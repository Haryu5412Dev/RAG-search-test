"""
GUI 버전을 EXE 파일로 빌드하는 스크립트
PyInstaller를 사용하여 단일 실행 파일 생성
"""

import os
import sys
import shutil
from pathlib import Path


def build_exe():
    """PyInstaller를 사용하여 EXE 파일 생성"""
    
    print("=" * 60)
    print("RAG 검색 시스템 GUI 빌드 시작")
    print("=" * 60)
    
    # PyInstaller 설치 확인
    try:
        import PyInstaller
        print("✓ PyInstaller 설치 확인됨")
    except ImportError:
        print("✗ PyInstaller가 설치되어 있지 않습니다.")
        print("  다음 명령어로 설치하세요: pip install pyinstaller")
        sys.exit(1)
    
    # 필요한 파일/폴더 확인
    required_paths = [
        "gui.py",
        "util/",
        "data/PLM_training_manual_clean.txt"
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            print(f"✗ 필수 파일/폴더를 찾을 수 없습니다: {path}")
            sys.exit(1)
    
    print("✓ 필수 파일 확인 완료")
    
    # 빌드 명령어 구성
    build_command = [
        "pyinstaller",
        "--name=RAG검색시스템",          # 실행 파일 이름
        "--onefile",                      # 단일 파일로 패키징
        "--windowed",                     # 콘솔 창 숨기기 (GUI만 표시)
        "--icon=NONE",                    # 아이콘 (필요시 .ico 파일 경로 지정)
        
        # 데이터 파일 포함
        "--add-data=data;data",           # data 폴더 포함
        "--add-data=.env;.",              # .env 파일 포함 (있는 경우)
        
        # 숨겨진 import 명시
        "--hidden-import=tiktoken_ext.openai_public",
        "--hidden-import=tiktoken_ext",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors.typedefs",
        "--hidden-import=sklearn.neighbors.quad_tree",
        "--hidden-import=sklearn.tree._utils",
        
        # 제외할 모듈 (용량 감소)
        "--exclude-module=matplotlib",
        "--exclude-module=pandas",
        "--exclude-module=PIL",
        "--exclude-module=PyQt5",
        
        # 진입점
        "gui.py"
    ]
    
    print("\n빌드 명령어:")
    print(" ".join(build_command))
    print("\n빌드 시작... (5-10분 소요될 수 있습니다)")
    print("-" * 60)
    
    # 빌드 실행
    os.system(" ".join(build_command))
    
    print("-" * 60)
    print("\n빌드 완료!")
    
    # 결과 확인
    exe_path = Path("dist/RAG검색시스템.exe")
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"✓ 실행 파일 생성됨: {exe_path}")
        print(f"  파일 크기: {size_mb:.1f} MB")
        print(f"\n실행 방법:")
        print(f"  1. {exe_path} 파일을 더블클릭")
        print(f"  2. data 폴더와 .env 파일이 같은 위치에 있어야 함")
    else:
        print("✗ 빌드 실패: 실행 파일을 찾을 수 없습니다.")
        print("  build/ 폴더의 로그를 확인하세요.")
    
    print("\n" + "=" * 60)


def clean_build():
    """빌드 임시 파일 정리"""
    print("\n임시 파일 정리 중...")
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    files_to_clean = ["*.spec"]
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"  삭제됨: {dir_name}/")
    
    import glob
    for pattern in files_to_clean:
        for file in glob.glob(pattern):
            Path(file).unlink()
            print(f"  삭제됨: {file}")
    
    print("정리 완료!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG GUI 빌드 스크립트")
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="빌드 후 임시 파일 정리"
    )
    
    args = parser.parse_args()
    
    try:
        build_exe()
        
        if args.clean:
            clean_build()
    
    except KeyboardInterrupt:
        print("\n\n빌드 취소됨.")
        sys.exit(1)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)
