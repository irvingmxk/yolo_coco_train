#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天记录保存和合并功能使用示例
演示如何使用 bubble_ocr.py 的聊天记录保存和智能合并功能
"""

from bubble_ocr import process_image, load_yolo_model, init_ocr_engine, CONFIG
from chat_record_manager import chat_record_manager
from pathlib import Path


def example_single_image():
    """示例1: 处理单张图片（第一次，直接保存）"""
    print("\n" + "="*60)
    print("示例1: 处理第一张图片")
    print("="*60)
    
    # 加载模型
    yolo_model = load_yolo_model(CONFIG['yolo_model'])
    ocr_engine = init_ocr_engine(
        backend=CONFIG['ocr_backend'],
        lang=CONFIG['ocr_lang']
    )
    
    # 处理第一张图片
    image_path = "/workspace/yolo/image/tinder2.jpg"
    session_id = "test_session_001"
    
    result = process_image(image_path, yolo_model, ocr_engine, CONFIG, session_id=session_id)
    
    # 查看合并信息
    if result.get('merge_info'):
        merge_info = result['merge_info']
        print(f"\n✅ 处理完成!")
        print(f"   操作类型: {merge_info['operation_type']}")
        print(f"   新增消息: {merge_info['added_count']} 条")
        print(f"   累计总数: {merge_info['total_count']} 条")
    
    return session_id, result


def example_merge_images(session_id: str):
    """示例2: 处理第二张图片（智能合并）"""
    print("\n" + "="*60)
    print("示例2: 处理第二张图片（智能合并）")
    print("="*60)
    
    # 加载模型
    yolo_model = load_yolo_model(CONFIG['yolo_model'])
    ocr_engine = init_ocr_engine(
        backend=CONFIG['ocr_backend'],
        lang=CONFIG['ocr_lang']
    )
    
    # 使用相同的 session_id 处理第二张图片
    # 假设这是同一聊天会话的下一页截图
    image_path = "/workspace/yolo/image/tinder2.jpg"  # 实际使用时替换为第二张图片
    
    result = process_image(image_path, yolo_model, ocr_engine, CONFIG, session_id=session_id)
    
    # 查看合并信息
    if result.get('merge_info'):
        merge_info = result['merge_info']
        print(f"\n✅ 合并完成!")
        print(f"   操作类型: {merge_info['operation_type']}")
        print(f"   新增消息: {merge_info['added_count']} 条")
        print(f"   累计总数: {merge_info['total_count']} 条")
    
    return result


def example_get_chat_records(session_id: str):
    """示例3: 获取保存的聊天记录"""
    print("\n" + "="*60)
    print("示例3: 获取保存的聊天记录")
    print("="*60)
    
    # 获取文本格式
    chat_text = chat_record_manager.get_chat_text(session_id)
    print(f"\n聊天记录文本格式:\n{chat_text}\n")
    
    # 获取结构化数据
    records = chat_record_manager.get_chat_records(session_id)
    print(f"共 {len(records)} 条消息:")
    for i, record in enumerate(records, 1):
        print(f"  {i}. [{record['speaker']}] {record['content']}")


def example_different_sessions():
    """示例4: 不同会话使用不同的 session_id"""
    print("\n" + "="*60)
    print("示例4: 不同会话独立保存")
    print("="*60)
    
    # 加载模型
    yolo_model = load_yolo_model(CONFIG['yolo_model'])
    ocr_engine = init_ocr_engine(
        backend=CONFIG['ocr_backend'],
        lang=CONFIG['ocr_lang']
    )
    
    # 会话1
    session_id_1 = "chat_session_A"
    image_path_1 = "/workspace/yolo/image/tinder2.jpg"
    result_1 = process_image(image_path_1, yolo_model, ocr_engine, CONFIG, session_id=session_id_1)
    
    # 会话2（独立的会话）
    session_id_2 = "chat_session_B"
    image_path_2 = "/workspace/yolo/image/tinder2.jpg"
    result_2 = process_image(image_path_2, yolo_model, ocr_engine, CONFIG, session_id=session_id_2)
    
    print(f"\n会话 {session_id_1} 有 {len(chat_record_manager.get_chat_records(session_id_1))} 条消息")
    print(f"会话 {session_id_2} 有 {len(chat_record_manager.get_chat_records(session_id_2))} 条消息")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("聊天记录保存和合并功能示例")
    print("="*60)
    
    # 示例1: 处理第一张图片
    session_id, result1 = example_single_image()
    
    # 示例2: 处理第二张图片（智能合并）
    # result2 = example_merge_images(session_id)
    
    # 示例3: 获取保存的聊天记录
    example_get_chat_records(session_id)
    
    # 示例4: 不同会话独立保存
    # example_different_sessions()
    
    print("\n" + "="*60)
    print("示例完成!")
    print("="*60)

