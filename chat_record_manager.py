#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天记录管理和智能合并模块
用于 OCR 识别的聊天气泡记录保存和合并
"""

import os
import json
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path


class ChatRecordManager:
    """聊天记录管理器，负责保存和智能合并聊天记录"""
    
    def __init__(self, base_dir: str = "./chat_records", similarity_threshold: float = 0.85):
        """
        初始化聊天记录管理器
        
        Args:
            base_dir: 聊天记录保存的基础目录
            similarity_threshold: 消息相似度阈值（用于合并判断）
        """
        self.base_dir = base_dir
        self.similarity_threshold = similarity_threshold
        # 用于存储每个会话的累积记录（内存缓存）
        self.accumulated_records: Dict[str, List[Dict]] = {}
        # 确保基础目录存在
        Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    def normalize_message(self, content: str) -> str:
        """
        消息标准化，用于生成签名
        """
        if not content:
            return ""
        
        # Unicode规范化
        s = unicodedata.normalize("NFKC", content)
        
        # 统一引号和括号
        trans = str.maketrans({
            "（": "(",
            "）": ")",
            '"': '"',
            '"': '"',
            "'": "'",
            "'": "'",
        })
        s = s.translate(trans)
        
        # 统一空白字符为单个空格
        s = re.sub(r"\s+", " ", s).strip()
        
        return s
    
    def canon_text(self, s: str) -> str:
        """用于模糊匹配的文本规范化"""
        if not s:
            return ""
        
        t = unicodedata.normalize("NFKC", s)
        
        # 统一标点
        t = (
            t.replace("，", ",")
            .replace("。", ".")
            .replace("？", "?")
            .replace("！", "!")
        )
        
        # 转小写，统一空白
        t = re.sub(r"\s+", " ", t).strip().lower()
        
        return t
    
    def get_similarity_threshold(self, text1: str, text2: str) -> float:
        """根据文本长度动态调整相似度阈值"""
        min_len = min(len(text1), len(text2))
        
        # OCR场景中，短文本的识别错误影响更大，适当降低阈值
        if min_len < 10:
            return 0.80  # 短文本
        elif min_len < 30:
            return 0.82  # 中等文本
        else:
            return self.similarity_threshold  # 长文本使用默认阈值
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（使用 SequenceMatcher）"""
        if not text1 or not text2:
            return 0.0
        
        # 使用规范化后的文本比较
        canon1 = self.canon_text(text1)
        canon2 = self.canon_text(text2)
        
        return SequenceMatcher(None, canon1, canon2).ratio()
    
    def fuzzy_equal(self, msg1: Dict, msg2: Dict) -> bool:
        """判断两条消息是否等价（考虑OCR错误）"""
        speaker1 = msg1.get("speaker", "").strip()
        speaker2 = msg2.get("speaker", "").strip()
        
        # 说话者必须相同
        if speaker1 != speaker2:
            return False
        
        content1 = (msg1.get("content") or "").strip()
        content2 = (msg2.get("content") or "").strip()
        
        # 快速路径：完全相等
        if content1 == content2:
            return True
        
        # 规范化后完全相等
        norm1 = self.normalize_message(content1)
        norm2 = self.normalize_message(content2)
        if norm1 == norm2:
            return True
        
        # 使用相似度判断（考虑OCR错误）
        threshold = self.get_similarity_threshold(content1, content2)
        similarity = self.calculate_text_similarity(content1, content2)
        
        return similarity >= threshold
    
    def _are_records_identical(self, records1: List[Dict], records2: List[Dict]) -> bool:
        """快速检测两个记录列表是否完全相同"""
        if len(records1) != len(records2):
            return False
        
        for r1, r2 in zip(records1, records2):
            if not isinstance(r1, dict) or not isinstance(r2, dict):
                return False
            
            speaker1 = (r1.get("speaker") or "").strip()
            speaker2 = (r2.get("speaker") or "").strip()
            content1 = (r1.get("content") or "").strip()
            content2 = (r2.get("content") or "").strip()
            
            if speaker1 != speaker2 or content1 != content2:
                return False
        
        return True
    
    def _is_subset(self, new_records: List[Dict], current_records: List[Dict]) -> bool:
        """检查新记录是否是当前记录的子集"""
        if not new_records or not current_records:
            return False
        
        if len(new_records) >= len(current_records):
            return False
        
        # 检查新记录的所有消息是否都在当前记录中
        for new_msg in new_records:
            found = False
            for current_msg in current_records:
                if self.fuzzy_equal(new_msg, current_msg):
                    found = True
                    break
            if not found:
                return False
        
        return True
    
    def merge_messages(self, existing_messages: List[Dict], new_messages: List[Dict]) -> Tuple[List[Dict], int, str]:
        """
        智能合并消息（使用LCS算法）
        
        Args:
            existing_messages: 已存在的消息列表
            new_messages: 新消息列表
            
        Returns:
            (合并后的消息列表, 新增消息数量, 操作类型)
        """
        if not new_messages:
            return existing_messages, 0, "no_new"
        
        if not existing_messages:
            return new_messages, len(new_messages), "initial"
        
        n, m = len(existing_messages), len(new_messages)
        
        # 动态规划表（LCS）
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # 填充DP表
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if self.fuzzy_equal(existing_messages[i], new_messages[j]):
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        
        # 回溯构建合并结果
        merged = []
        added_count = 0
        i = j = 0
        
        while i < n and j < m:
            if self.fuzzy_equal(existing_messages[i], new_messages[j]):
                # 消息匹配，选择更长的版本（可能是OCR识别更准确）
                existing_content = existing_messages[i].get("content", "")
                new_content = new_messages[j].get("content", "")
                if len(new_content) > len(existing_content):
                    merged.append(new_messages[j])
                else:
                    merged.append(existing_messages[i])
                i += 1
                j += 1
            elif dp[i + 1][j] >= dp[i][j + 1]:
                # 保留existing消息
                merged.append(existing_messages[i])
                i += 1
            else:
                # 添加new消息
                merged.append(new_messages[j])
                added_count += 1
                j += 1
        
        # 处理剩余消息
        while i < n:
            merged.append(existing_messages[i])
            i += 1
        while j < m:
            merged.append(new_messages[j])
            added_count += 1
            j += 1
        
        return merged, added_count, "global_align_merge"
    
    def merge_and_save(self, session_id: str, new_records: List[Dict]) -> Tuple[int, int, str]:
        """
        智能合并并保存聊天记录
        
        Args:
            session_id: 会话ID（用于标识不同的聊天会话）
            new_records: 新的聊天记录列表，格式: [{"speaker": "user", "content": "..."}, ...]
            
        Returns:
            (新增消息数量, 总消息数量, 操作类型)
        """
        try:
            # 获取当前累积的记录（先从内存，如果没有则从文件加载）
            if session_id not in self.accumulated_records:
                # 尝试从文件加载
                self.accumulated_records[session_id] = self._load_from_file(session_id)
            
            current_records = self.accumulated_records.get(session_id, [])
            
            if not new_records:
                return 0, len(current_records), "no_new"
            
            if not current_records:
                # 第一次记录，直接保存
                self.accumulated_records[session_id] = new_records.copy()
                self._save_to_file(session_id, new_records)
                return len(new_records), len(new_records), "initial"
            
            # 快速检测：完全相同
            if self._are_records_identical(current_records, new_records):
                return 0, len(current_records), "identical_skip"
            
            # 快速检测：子集
            if self._is_subset(new_records, current_records):
                return 0, len(current_records), "subset_skip"
            
            # 智能合并
            merged_records, added_count, operation_type = self.merge_messages(
                current_records, new_records
            )
            
            # 更新内存记录
            self.accumulated_records[session_id] = merged_records
            
            # 保存到文件
            self._save_to_file(session_id, merged_records)
            
            return added_count, len(merged_records), operation_type
        
        except Exception as e:
            print(f"❌ 合并保存失败 [{session_id}]: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0, len(self.accumulated_records.get(session_id, [])), "error"
    
    def _load_from_file(self, session_id: str) -> List[Dict]:
        """从文件加载聊天记录"""
        chat_file_path = os.path.join(self.base_dir, session_id, "chat_records.txt")
        
        if not os.path.exists(chat_file_path):
            return []
        
        try:
            records = []
            with open(chat_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 解析格式: "speaker: content"
                    if ":" in line:
                        parts = line.split(":", 1)
                        speaker = parts[0].strip()
                        content = parts[1].strip() if len(parts) > 1 else ""
                        if speaker and content:
                            records.append({"speaker": speaker, "content": content})
            return records
        except Exception as e:
            print(f"⚠️ 加载聊天记录失败 [{session_id}]: {str(e)}")
            return []
    
    def _save_to_file(self, session_id: str, records: List[Dict]):
        """
        保存记录到文件（原子写入）
        
        Args:
            session_id: 会话ID
            records: 聊天记录列表
        """
        try:
            # 创建会话目录
            dir_path = os.path.join(self.base_dir, session_id)
            os.makedirs(dir_path, exist_ok=True)
            
            # 保存到文件
            chat_file = os.path.join(dir_path, "chat_records.txt")
            tmp_file = chat_file + ".tmp"
            
            # 先写入临时文件
            with open(tmp_file, "w", encoding="utf-8") as f:
                if records and isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            speaker = record.get("speaker", "")
                            content = record.get("content", "")
                            if speaker and content:
                                f.write(f"{speaker}: {content}\n")
                        else:
                            f.write(f"unknown: {str(record)}\n")
            
            # 原子替换
            if os.path.exists(tmp_file):
                os.replace(tmp_file, chat_file)
            
            # 同时保存JSON格式（便于后续处理）
            json_file = os.path.join(dir_path, "chat_records.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": session_id,
                    "messages": records,
                    "total_count": len(records),
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"❌ 保存聊天记录失败 [{session_id}]: {str(e)}")
            # 清理临时文件
            tmp_file = os.path.join(self.base_dir, session_id, "chat_records.txt.tmp")
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass
    
    def get_chat_records(self, session_id: str) -> List[Dict]:
        """获取指定会话的聊天记录"""
        # 先从内存获取，如果没有则从文件加载
        if session_id not in self.accumulated_records:
            self.accumulated_records[session_id] = self._load_from_file(session_id)
        
        return self.accumulated_records.get(session_id, [])
    
    def clear_records(self, session_id: str):
        """清空指定会话的记录"""
        if session_id in self.accumulated_records:
            del self.accumulated_records[session_id]
        
        # 清空文件
        self._save_to_file(session_id, [])
    
    def get_chat_text(self, session_id: str) -> str:
        """获取聊天记录的文本格式"""
        records = self.get_chat_records(session_id)
        lines = [f"{r.get('speaker', '')}: {r.get('content', '')}" for r in records]
        return "\n".join(lines)


# 全局实例
chat_record_manager = ChatRecordManager()

