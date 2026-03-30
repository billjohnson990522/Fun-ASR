#!/usr/bin/env python3
"""
Fun-ASR 语音识别演示脚本
支持中文、英文、日文等多种语言的语音识别

音频文件存放位置：
- 默认目录：backend/fun-asr/audio/
- 也可以使用绝对路径指定任意位置的音频文件

使用方法：
  python asr_demo.py                           # 使用模型自带示例
  python asr_demo.py audio/test.mp3            # 使用 audio 目录下的文件
  python asr_demo.py /path/to/audio.mp3        # 使用绝对路径
  python asr_demo.py audio/test.mp3 英文       # 指定语言
"""

from funasr import AutoModel
import os
import sys

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 默认音频目录
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")


def create_model(model_dir: str = "FunAudioLLM/Fun-ASR-Nano-2512", device: str = "cpu"):
    """
    创建并返回 ASR 模型
    
    Args:
        model_dir: 模型名称或本地路径
        device: 运行设备，如 "cuda:0" 或 "cpu"
    
    Returns:
        AutoModel 实例
    """
    print(f"正在加载模型: {model_dir}")
    print(f"使用设备: {device}")
    
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
        hub="ms"  # 使用 ModelScope
    )
    
    print("模型加载完成!")
    return model


def transcribe(model, audio_path: str, language: str = "中文", hotwords: list = None, use_vad: bool = False):
    """
    对音频文件进行语音识别
    
    Args:
        model: AutoModel 实例
        audio_path: 音频文件路径
        language: 语言类型，支持 "中文"、"英文"、"日文"
        hotwords: 热词列表，提高特定词汇的识别准确率
        use_vad: 是否使用 VAD（语音活动检测）
    
    Returns:
        识别出的文本
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    if hotwords is None:
        hotwords = []
    
    print(f"\n正在识别音频: {audio_path}")
    print(f"语言: {language}")
    if hotwords:
        print(f"热词: {hotwords}")
    
    if use_vad:
        # 使用 VAD 进行长音频分段处理
        vad_model = AutoModel(
            model=model.model_path if hasattr(model, 'model_path') else "FunAudioLLM/Fun-ASR-Nano-2512",
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            remote_code="./model.py",
            device=model.device if hasattr(model, 'device') else "cpu",
        )
        res = vad_model.generate(
            input=[audio_path],
            cache={},
            batch_size=1
        )
    else:
        res = model.generate(
            input=[audio_path],
            cache={},
            batch_size=1,
            hotwords=hotwords,
            language=language,
            itn=True,  # 启用逆文本标准化
        )
    
    if res and len(res) > 0:
        text = res[0]["text"]
        return text
    return ""


def resolve_audio_path(audio_input: str) -> str:
    """
    解析音频路径，支持相对路径和绝对路径
    
    Args:
        audio_input: 用户输入的音频路径
    
    Returns:
        解析后的绝对路径
    """
    # 如果是绝对路径，直接返回
    if os.path.isabs(audio_input):
        return audio_input
    
    # 检查相对于当前工作目录
    cwd_path = os.path.abspath(audio_input)
    if os.path.exists(cwd_path):
        return cwd_path
    
    # 检查相对于脚本目录
    script_path = os.path.join(SCRIPT_DIR, audio_input)
    if os.path.exists(script_path):
        return script_path
    
    # 检查 audio 目录下
    audio_dir_path = os.path.join(AUDIO_DIR, audio_input)
    if os.path.exists(audio_dir_path):
        return audio_dir_path
    
    # 返回原始路径（会在后续检查时报错）
    return audio_input


def main():
    """主函数"""
    print("="*60)
    print("Fun-ASR 语音识别系统")
    print("="*60)
    print(f"音频文件目录: {AUDIO_DIR}")
    print(f"请将音频文件放入该目录，或使用绝对路径指定")
    print("="*60)
    
    # 配置参数
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    device = "cpu"  # 如果有 GPU，可以改为 "cuda:0"
    
    # 创建模型
    model = create_model(model_dir=model_dir, device=device)
    
    # 如果用户提供了音频路径
    if len(sys.argv) > 1:
        audio_input = sys.argv[1]
        audio_path = resolve_audio_path(audio_input)
        language = sys.argv[2] if len(sys.argv) > 2 else "中文"
        
        print("\n" + "="*50)
        print(f"识别音频: {audio_path}")
        print(f"语言: {language}")
        print("="*50)
        
        text = transcribe(model, audio_path, language=language)
        print(f"\n识别结果: {text}")
    else:
        # 使用模型自带的示例音频
        example_audio = f"{model.model_path}/example/zh.mp3"
        
        if os.path.exists(example_audio):
            print("\n" + "="*50)
            print("识别模型自带的示例音频")
            print("="*50)
            
            text = transcribe(
                model, 
                example_audio, 
                language="中文",
                hotwords=["开放时间"]
            )
            print(f"\n识别结果: {text}")
        else:
            print(f"\n示例音频不存在，请将音频文件放入: {AUDIO_DIR}")
            print("\n使用方法:")
            print("  python asr_demo.py your_audio.mp3        # 识别 audio 目录下的文件")
            print("  python asr_demo.py /path/to/audio.mp3    # 使用绝对路径")
            print("  python asr_demo.py audio.mp3 英文        # 指定语言")


if __name__ == "__main__":
    main()
