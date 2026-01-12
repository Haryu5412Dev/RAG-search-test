"""
RAG 통합 프로그램
문서 검색 → Context 구성 → LLM 요약 답변 생성 → 출력

자동으로 청크를 생성하고, data/PLM_training_manual_clean.txt를 사용하여
질문에 대한 답변을 제공합니다.
"""

from __future__ import annotations

import sys
import threading
import time
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
import questionary
from questionary import Style

from util.chunking import load_chunks
from util.generation import answer_with_llm, build_context
from util.retrieval import TfidfRetriever


# 커스텀 스타일 정의
custom_style = Style([
    ('qmark', 'fg:#e5c07b bold'),           # 물음표 - 황금색
    ('question', 'fg:#61afef bold'),        # 질문 텍스트 - 파란색
    ('answer', 'fg:#98c379 bold'),          # 선택된 답변 - 초록색
    ('pointer', 'fg:#e5c07b bold'),         # 포인터 (>) - 황금색
    ('highlighted', 'fg:#e5c07b bold'),     # 하이라이트된 선택지 - 황금색
    ('selected', 'fg:#98c379'),             # 선택된 항목 - 초록색
    ('separator', 'fg:#6c6c6c'),            # 구분선 - 회색
    ('instruction', 'fg:#abb2bf'),          # 설명 - 밝은 회색
    ('text', 'fg:#abb2bf'),                 # 일반 텍스트 - 밝은 회색
])


class LoadingSpinner:
    """로딩 스피너를 표시하는 헬퍼 클래스"""
    
    def __init__(self, message: str = "처리 중"):
        self.message = message
        self.is_running = False
        self.thread = None
    
    def _spin(self):
        """스피너 애니메이션"""
        spinner = ['|', '/', '-', '\\']
        idx = 0
        while self.is_running:
            print(f'\r[>] {self.message}... {spinner[idx]}', end='', flush=True)
            idx = (idx + 1) % len(spinner)
            time.sleep(0.2)
        print('\r' + ' ' * (len(self.message) + 20) + '\r', end='', flush=True)
    
    def start(self):
        """스피너 시작"""
        self.is_running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
    
    def stop(self):
        """스피너 중지"""
        self.is_running = False
        if self.thread:
            self.thread.join()


class RAGSystem:
    """통합 RAG 시스템"""

    def __init__(self, document_path: str = "data/PLM_training_manual_clean.txt"):
        """
        RAG 시스템 초기화
        
        Args:
            document_path: 문서 파일 경로
        """
        self.document_path = Path(document_path)
        self.retriever: Optional[TfidfRetriever] = None
        self.chunks: list[dict] = []
        self._initialized = False
        
        # output 폴더 생성
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def clear_screen(self):
        """터미널 화면 청소"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def initialize(self) -> bool:
        """
        시스템 초기화: 문서 로드 → 청크 생성 → 검색기 준비
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            print("\n" + "=" * 60)
            print("  RAG 시스템 초기화 중...")
            print("=" * 60)

            # 1. 문서 파일 확인
            if not self.document_path.exists():
                print(f"[X] 오류: 문서 파일을 찾을 수 없습니다: {self.document_path}")
                return False

            print(f"[v] 문서 파일 로드: {self.document_path}")

            # 2. 청크 생성
            print("[v] 청크 생성 중...")
            self.chunks = load_chunks(self.document_path)
            
            if not self.chunks:
                print("[X] 오류: 유효한 청크를 생성하지 못했습니다.")
                print("    문서 포맷을 확인하세요 (예: [페이지 n] 헤더 형식)")
                return False

            print(f"[v] 총 {len(self.chunks)}개의 청크 생성 완료")

            # 3. 검색기 초기화
            print("[v] 검색 인덱스 구축 중...")
            self.retriever = TfidfRetriever()
            self.retriever.fit(self.chunks)
            print("[v] 검색 인덱스 구축 완료")

            # 4. LLM 모델 준비 (첫 호출 시 로드되므로 여기서는 안내만)
            print("[v] LLM 모델: Qwen3-4B-Instruct")
            print("    (i) 첫 질문 시 모델 로드에 30초~1분 정도 소요될 수 있습니다.")
            print("[OK] 초기화 완료!\n")

            self._initialized = True
            return True

        except Exception as e:
            print(f"[X] 초기화 중 오류 발생: {type(e).__name__}: {e}")
            return False

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        """
        질문에 대한 문서 검색
        
        Args:
            question: 사용자 질문
            top_k: 반환할 검색 결과 개수
            
        Returns:
            list[dict]: 검색된 청크 리스트
        """
        if not self._initialized or self.retriever is None:
            raise RuntimeError("RAG 시스템이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        hits = self.retriever.search(question, top_k=top_k)
        return [hit.chunk for hit in hits]

    def save_to_file(self, question: str, search_results: list[dict], answer: str) -> Path:
        """
        검색 결과와 답변을 파일로 저장
        
        Args:
            question: 질문
            search_results: 검색 결과
            answer: LLM 답변
            
        Returns:
            Path: 저장된 파일 경로
        """
        # 파일명 생성: 질문요약_(년월일_시분초).txt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 질문을 파일명에 적합하게 변환 (최대 20자)
        safe_question = "".join(c if c.isalnum() or c in " _-" else "_" for c in question[:20])
        safe_question = safe_question.strip().replace(" ", "_")
        filename = f"{safe_question}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        # 파일 내용 작성
        content = []
        content.append("=" * 70)
        content.append("RAG 질의응답 결과")
        content.append("=" * 70)
        content.append(f"\n생성 시각: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
        content.append(f"\n[질문]\n{question}\n")
        
        # 검색 결과
        content.append("=" * 70)
        content.append(f"[검색 결과] Top-{len(search_results)}")
        content.append("=" * 70)
        for rank, chunk in enumerate(search_results, 1):
            page = chunk.get("page", "?")
            header = (chunk.get("header") or "").strip()
            text = (chunk.get("text") or "").strip()
            
            content.append(f"\n[{rank}] 페이지 {page}")
            if header:
                content.append(f"제목: {header}")
            content.append(f"\n내용:\n{text}\n")
            content.append("-" * 70)
        
        # LLM 답변
        content.append("\n" + "=" * 70)
        content.append("[LLM 답변]")
        content.append("=" * 70)
        content.append(f"\n{answer}\n")
        content.append("=" * 70)
        
        # 파일 저장
        filepath.write_text("\n".join(content), encoding="utf-8")
        
        return filepath

    def answer_question(self, question: str, top_k: int = 3, show_context: bool = True) -> str:
        """
        질문에 대한 답변 생성 (전체 흐름)
        
        Args:
            question: 사용자 질문
            top_k: 검색할 문서 개수
            show_context: 검색 결과를 출력할지 여부
            
        Returns:
            str: LLM 생성 답변
        """
        if not self._initialized:
            raise RuntimeError("RAG 시스템이 초기화되지 않았습니다.")

        try:
            # 1. 문서 검색
            print(f"\n[?] 질문: {question}\n")
            
            search_results = self.search(question, top_k=top_k)
            
            if not search_results:
                return "[X] 검색 결과가 없습니다. 다른 질문을 시도해보세요."

            # 2. 검색 결과 출력 (Top-3 요약)
            if show_context:
                print(f"[검색 결과] Top-{len(search_results)}:")
                for rank, chunk in enumerate(search_results, 1):
                    page = chunk.get("page", "?")
                    header = (chunk.get("header") or "").strip()
                    text = (chunk.get("text") or "").strip()
                    snippet = " ".join(text.split())[:150] + "..." if len(text) > 150 else text
                    
                    print(f"  [{rank}] 페이지 {page}", end="")
                    if header:
                        print(f" - {header}")
                    else:
                        print()
                    print(f"      {snippet}")
                print()

            # 3. Context 구성
            context = build_context(search_results, max_chunks=top_k)
            
            if not context.strip():
                return "[X] 문서에 관련 내용이 없습니다."

            # 4. LLM 답변 생성
            print("[AI] 답변 생성 중...")
            print("     (!) Qwen 모델이 답변을 생성하고 있습니다.")
            print("     (i) 첫 실행 시 모델 로드에 시간이 걸릴 수 있습니다.\n")
            
            # 로딩 스피너 시작
            spinner = LoadingSpinner("답변 생성 중")
            spinner.start()
            
            try:
                answer = answer_with_llm(question, context)
            finally:
                spinner.stop()

            # 5. 답변 출력
            print("[답변]")
            print(answer)
            print()
            
            # 6. 파일 저장
            try:
                saved_file = self.save_to_file(question, search_results, answer)
                print(f"[저장 완료] 결과가 저장되었습니다")
                print(f"           파일: output/{saved_file.name}\n")
            except Exception as e:
                print(f"[!] 파일 저장 중 오류: {e}\n")

            return answer

        except RuntimeError as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                print(f"\n[X] API 요청 제한 오류: {error_msg}")
                return "[X] API 요청 제한에 도달했습니다. 잠시 후 다시 시도하세요."
            elif "api key" in error_msg.lower():
                print(f"\n[X] API 키 오류: {error_msg}")
                return "[X] API 키가 설정되지 않았습니다. 환경변수를 확인하세요."
            else:
                print(f"\n[X] 오류 발생: {error_msg}")
                return f"[X] 오류: {error_msg}"
        
        except Exception as e:
            print(f"\n[X] 예상치 못한 오류: {type(e).__name__}: {e}")
            return f"[X] 시스템 오류: {type(e).__name__}"

    def show_menu(self) -> str:
        """메뉴 출력 및 선택 입력 (화살표 키 + 숫자 키)"""
        menu_choice = questionary.select(
            "원하는 작업을 선택하세요:",
            choices=[
                "질문하기",
                "프로그램 종료"
            ],
            style=custom_style,
            use_shortcuts=True,
            use_arrow_keys=True,
            instruction="(화살표 키 ↑↓ 또는 숫자 키로 선택, Enter로 확인)"
        ).ask()
        
        if menu_choice is None:  # Ctrl+C 등으로 취소
            return "2"
        
        # 선택을 번호로 변환
        if menu_choice == "질문하기":
            return "1"
        elif menu_choice == "프로그램 종료":
            return "2"
        
        return "2"

    def run_interactive(self) -> None:
        """대화형 인터페이스 실행"""
        if not self._initialized:
            print("먼저 시스템을 초기화하세요.")
            return

        while True:
            try:
                choice = self.show_menu()
                
                if choice == "1":
                    # 화면 청소
                    self.clear_screen()
                    
                    # 질문 입력
                    print()
                    question = questionary.text(
                        "질문을 입력하세요:",
                        style=custom_style,
                        instruction="(취소: Ctrl+C)"
                    ).ask()
                    
                    if not question or not question.strip():
                        print("[!] 질문이 취소되었습니다.\n")
                        input("Enter 키를 눌러 계속...")
                        self.clear_screen()
                        continue
                    
                    # 질문 처리 전 화면 청소
                    self.clear_screen()
                    
                    self.answer_question(question.strip(), top_k=3, show_context=False)
                    
                    # 답변 후 계속 여부 확인
                    questionary.press_any_key_to_continue(
                        "아무 키나 눌러 메뉴로 돌아가기...",
                        style=custom_style
                    ).ask()
                    
                    # 화면 청소
                    self.clear_screen()
                
                elif choice == "2":
                    self.clear_screen()
                    print("\n[BYE] 프로그램을 종료합니다.")
                    break

            except KeyboardInterrupt:
                self.clear_screen()
                print("\n\n[BYE] 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n[X] 오류 발생: {type(e).__name__}: {e}")
                print("다시 시도하세요.\n")
                questionary.press_any_key_to_continue("아무 키나 눌러 계속...").ask()
                self.clear_screen()


def main() -> None:
    """메인 함수"""
    # 환경변수 로드 (.env 파일)
    load_dotenv()

    # RAG 시스템 생성 및 초기화
    rag = RAGSystem(document_path="data/PLM_training_manual_clean.txt")
    
    if not rag.initialize():
        print("시스템 초기화 실패. 프로그램을 종료합니다.")
        sys.exit(1)

    # 대화형 인터페이스 실행
    rag.run_interactive()


if __name__ == "__main__":
    main()
