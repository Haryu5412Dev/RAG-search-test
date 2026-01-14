"""
RAG 통합 프로그램 - Modern GUI
Apple-style UI with Dark/Light Mode

CLI 기능을 유지하면서 세련된 GUI로 사용할 수 있습니다.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from util.chunking import load_chunks
from util.generation import answer_with_llm, build_context, preload_model
from util.retrieval import TfidfRetriever


# 모던 컬러 테마
class Theme:
    """라이트/다크 모드 테마 정의"""
    
    LIGHT = {
        'bg': '#FFFFFF',
        'fg': '#1D1D1F',
        'bg_secondary': '#F5F5F7',
        'border': '#E5E5EA',
        'accent': '#007AFF',
        'accent_hover': '#0051D5',
        'success': '#34C759',
        'text_secondary': '#86868B',
        'input_bg': '#FAFAFA',
        'card_bg': '#FFFFFF',
        'hover_bg': '#F0F0F5',
    }
    
    DARK = {
        'bg': '#000000',
        'fg': '#FFFFFF',
        'bg_secondary': '#1C1C1E',
        'border': '#38383A',
        'accent': '#0A84FF',
        'accent_hover': '#409CFF',
        'success': '#30D158',
        'text_secondary': '#98989D',
        'input_bg': '#1C1C1E',
        'card_bg': '#2C2C2E',
        'hover_bg': '#2C2C2E',
    }


class ModernText(tk.Text):
    """모던 스타일 텍스트 위젯"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.config(
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            padx=15,
            pady=12,
            font=('맑은 고딕', 10),
            wrap=tk.WORD,
            insertwidth=2,
        )


class LoadingOverlay(tk.Frame):
    """로딩 오버레이 (답변 생성 중 표시)"""
    
    def __init__(self, parent):
        super().__init__(parent, bg='#E5E5E5')
        self.place_forget()
        self.parent = parent
        
        # 중앙 카드 (그림자 효과를 위한 외곽 프레임)
        shadow_frame = tk.Frame(self, bg='#D0D0D0')
        shadow_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # 실제 카드
        self.card = tk.Frame(shadow_frame, bg='#FFFFFF', relief=tk.FLAT, 
                            borderwidth=1, highlightthickness=0)
        self.card.pack(padx=2, pady=2)
        
        # 상단 여백
        tk.Frame(self.card, bg='#FFFFFF', height=30).pack()
        
        # 로딩 애니메이션 (간단한 점 3개)
        self.loading_label = tk.Label(
            self.card,
            text="●  ●  ●",
            font=('맑은 고딕', 20),
            bg='#FFFFFF',
            fg='#007AFF',
            padx=50,
            pady=15
        )
        self.loading_label.pack()
        
        # 상태 메시지
        self.status_label = tk.Label(
            self.card,
            text="AI 답변 생성 중...",
            font=('맑은 고딕', 11),
            bg='#FFFFFF',
            fg='#1D1D1F'
        )
        self.status_label.pack(pady=(5, 30))
        
        self.animation_running = False
        self.animation_state = 0
    
    def show(self, message="AI 답변 생성 중..."):
        """오버레이 표시"""
        self.status_label.config(text=message)
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()
        self.animation_running = True
        self.animate()
    
    def hide(self):
        """오버레이 숨김"""
        self.animation_running = False
        self.place_forget()
    
    def animate(self):
        """로딩 애니메이션"""
        if not self.animation_running:
            return
        
        dots = ["●  ○  ○", "○  ●  ○", "○  ○  ●", "○  ●  ○"]
        self.loading_label.config(text=dots[self.animation_state % 4])
        self.animation_state += 1
        self.after(300, self.animate)
    
    def update_theme(self, theme):
        """테마 변경 시 색상 업데이트"""
        self.config(bg=theme['bg_secondary'])
        self.card.config(bg=theme['card_bg'])
        self.loading_label.config(bg=theme['card_bg'], fg=theme['accent'])
        self.status_label.config(bg=theme['card_bg'], fg=theme['fg'])


class RAGSystemModernGUI:
    """모던 RAG 시스템 GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RAG 문서 검색")
        self.root.geometry("1000x750")
        self.root.minsize(900, 600)
        
        # 환경 변수 로드
        load_dotenv()
        
        # 다크 모드 상태
        self.is_dark_mode = False
        self.current_theme = Theme.LIGHT
        
        # 시스템 변수
        self.document_path = Path("data/PLM_training_manual_clean.txt")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.retriever: Optional[TfidfRetriever] = None
        self.chunks: list[dict] = []
        self.initialized = False
        self.current_question = ""
        self.current_answer = ""
        
        # UI 구성
        self.setup_ui()
        
        # 초기화
        self.root.after(100, self.auto_initialize)
    
    def setup_ui(self):
        """UI 구성"""
        # 메인 컨테이너
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # 헤더
        self.setup_header()
        
        # 구분선
        self.header_separator = tk.Frame(self.main_container, height=1)
        self.header_separator.pack(fill=tk.X, padx=20)
        
        # 컨텐츠 (스크롤 가능)
        content = tk.Frame(self.main_container)
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=25)
        
        # 질문 영역
        self.setup_question_area(content)
        
        # 버튼 영역
        self.setup_buttons(content)
        
        # 검색 문맥 영역
        self.setup_context_area(content)
        
        # 답변 영역
        self.setup_answer_area(content)
        
        # 로딩 오버레이
        self.loading_overlay = LoadingOverlay(self.main_container)
        
        # 푸터
        self.setup_footer()
        
        # 초기 테마 적용
        self.apply_theme()
    
    def setup_header(self):
        """헤더 영역"""
        header = tk.Frame(self.main_container, height=70)
        header.pack(fill=tk.X, padx=30, pady=(25, 15))
        header.pack_propagate(False)
        
        # 타이틀
        title_frame = tk.Frame(header)
        title_frame.pack(side=tk.LEFT, anchor=tk.W)
        
        self.title_label = tk.Label(
            title_frame,
            text="RAG 문서 검색",
            font=('Segoe UI', 24, 'bold')
        )
        self.title_label.pack(anchor=tk.W)
        
        self.subtitle_label = tk.Label(
            title_frame,
            text="Qwen3-4B 기반 지능형 질의응답",
            font=('Segoe UI', 10)
        )
        self.subtitle_label.pack(anchor=tk.W, pady=(2, 0))
        
        # 우측 컨트롤
        right_frame = tk.Frame(header)
        right_frame.pack(side=tk.RIGHT, anchor=tk.E)
        
        # 상태 표시
        self.header_status = tk.Label(
            right_frame,
            text="🟡 초기화 중...",
            font=('맑은 고딕', 10)
        )
        self.header_status.pack(anchor=tk.E, pady=(0, 5))
        
        # 다크모드 토글 버튼
        self.theme_button = tk.Button(
            right_frame,
            text="🌙 다크 모드",
            font=('맑은 고딕', 9),
            command=self.toggle_theme,
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=6,
            borderwidth=1
        )
        self.theme_button.pack(anchor=tk.E)
    
    def setup_question_area(self, parent):
        """질문 입력 영역"""
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 20))
        
        label = tk.Label(
            frame,
            text="💬 질문 입력",
            font=('Segoe UI', 13, 'bold')
        )
        label.pack(anchor=tk.W, pady=(0, 10))
        
        # 텍스트 입력 컨테이너
        text_container = tk.Frame(frame, relief=tk.FLAT, borderwidth=1)
        text_container.pack(fill=tk.X)
        
        self.question_text = ModernText(text_container, height=4)
        self.question_text.pack(fill=tk.X, padx=1, pady=1)
    
    def setup_buttons(self, parent):
        """버튼 영역"""
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 25))
        
        # 메인 액션 버튼
        self.ask_btn = tk.Button(
            frame,
            text="🔍 질문하기",
            font=('맑은 고딕', 11, 'bold'),
            command=self.ask_question,
            relief=tk.FLAT,
            cursor='hand2',
            padx=30,
            pady=12,
            state=tk.DISABLED,
            borderwidth=0
        )
        self.ask_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        # 보조 버튼들
        self.clear_btn = tk.Button(
            frame,
            text="✕ 지우기",
            font=('맑은 고딕', 10),
            command=self.clear_question,
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=12,
            borderwidth=1
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.save_btn = tk.Button(
            frame,
            text="💾 저장",
            font=('맑은 고딕', 10),
            command=self.save_answer,
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=12,
            state=tk.DISABLED,
            borderwidth=1
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT)20))
        
        label = tk.Label(
            frame,
            text="📄 검색된 문맥",
            font=('Segoe UI', 12, 'bold')
        )
        label.pack(anchor=tk.W, pady=(0, 10))
        
        text_container = tk.Frame(frame, relief=tk.FLAT, borderwidth=1)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.context_text = ModernText(text_container, height=6, state=tk.DISABLED)
        self.context_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1
        )
        label.pack(anchor=tk.W, pady=(0, 8))
        
        self.context_text = ModernText(frame, height=5, state=tk.DISABLED)
        self.context_text.pack(fill=tk.BOTH, expand=True)
    ✨ AI 답변",
            font=('Segoe UI', 12, 'bold')
        )
        label.pack(anchor=tk.W, pady=(0, 10))
        
        text_container = tk.Frame(frame, relief=tk.FLAT, borderwidth=1)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.answer_text = ModernText(text_container, height=12, state=tk.DISABLED)
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1
            frame,
            text="답변",
            font=('맑은 고딕', 11, 'bold')
        )
        label.pack(anchor=tk.W, pady=(0, 8))
        # 구분선
        self.footer_separator = tk.Frame(self.main_container, height=1)
        self.footer_separator.pack(fill=tk.X, padx=30)
        
        footer = tk.Frame(self.main_container)
        footer.pack(fill=tk.X, padx=30, pady=(12, 20))
        
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        mode = "⚡ GPU" if use_gpu else "💻 CPU"
        
        self.footer_label = tk.Label(
            footer,
            text=f"{mode} 모드  •  Qwen3-4B",
            font=('Segoe UIenv("USE_GPU", "false").lower() == "true"
        mode = "GPU" if use_gpu else "CPU"
        
        self.footer_label = tk.Label(
            footer,
            text=f"실행 모드: {mode}",
            font=('맑은 고딕', 9)
        ) 모드" if self.is_dark_mode else "🌙 다크 모드
        self.footer_label.pack(side=tk.LEFT)
    
    def toggle_theme(self):
        """다크/라이트 모드 전환"""
        self.is_dark_mode = not self.is_dark_mode
        self.current_theme = Theme.DARK if self.is_dark_mode else Theme.LIGHT
        self.theme_button.config(text="☀️ 라이트" if self.is_dark_mode else "🌙 다크")
        self.apply_theme()
    
    def apply_theme(self):
        ""구분선
        self.header_separator.config(bg=theme['border'])
        self.footer_separator.config(bg=theme['border'])
        
        # 헤더
        for widget in [self.title_label, self.subtitle_label, self.header_status]:
            widget.config(bg=theme['bg'], fg=theme['fg'])
        
        self.subtitle_label.config(fg=theme['text_secondary'])
        
        # 텍스트 필드
        for text_widget in [self.question_text, self.context_text, self.answer_text]:
            text_widget.config(
                bg=theme['input_bg'],
                fg=theme['fg'],
                insertbackground=theme['accent'],
                highlightbackground=theme['border'],
                highlightcolor=theme['accent'],
                selectbackground=theme['accent'],
                selectforeground='#FFFFFF'
            )
        
        # 메인 버튼
        self.ask_btn.config(
            bg=theme['accent'],
            fg='#FFFFFF',
            activebackground=theme['accent_hover'],
            activeforeground='#FFFFFF'
        )
        
        # 보조 버튼
        for btn in [self.clear_btn, self.save_btn]:
            btn.config(
                bg=theme['bg'],
                fg=theme['fg'],
                activebackground=theme['hover_bg'],
                activeforeground=theme['fg'],
                highlightbackground=theme['border'],
                highlightcolor=theme['border']
            )
        
        # 테마 토글 버튼
        self.theme_button.config(
            bg=theme['bg'],
            fg=theme['fg'],
            activebackground=theme['hover_bg'],
            activeforeground=theme['fg'],
            highlightbackground=theme['border'],
            highlightcolor=theme['border']
        )
        
        # 푸터
        self.footer_label.config(bg=theme['bg'], fg=theme['text_secondary'])
        
        # 로딩 오버레이['fg'],
            activebackground=theme['hover_bg']
        )
        
        # 푸터
        self.footer_label.config(bg=theme['bg'], fg=theme['text_secondary'])
        
        # 로딩 오버레이 색상
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.update_theme(theme)
    
    def auto_initialize(self):
        """자동 초기화"""
        thread = threading.Thread(target=self.initialize_system, daemon=True)
        thread.start()
    
    def initialize_system(self):
        """시스템 초기화"""
        try:
            self.update_header_status("문서 로딩 중...", "warning")
            
            if not self.document_path.exists():
                self.show_error(f"문서를 찾을 수 없습니다: {self.document_path}")
                return
            
            self.update_header_status("문서 청킹 중...", "warning")
            self.chunks = load_chunks(str(self.document_path))
            
            if not self.chunks:
                self.show_error("청크를 생성할 수 없습니다.")
                return
            
            self.update_header_status("검색 엔진 초기화 중...", "warning")
            self.retriever = TfidfRetriever()
            self.retriever.fit(self.chunks)
            
            self.update_header_status("AI 모델 로딩 중...", "warning")
            preload_model()
            
            self.initialized = True
            self.update_header_status(f"준비 완료 ({len(self.chunks)}개 청크)", "success")
            
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.show_error(f"초기화 실패: {str(e)}")
            self.update_header_status("초기화 실패", "error")
    
    def update_header_status(self, message: str, status_type: str = "info"):
        """헤더 상태 업데이트"""
        colors = {
            "success": "🟢",
            "warning": "🟡",
            "error": "🔴",
            "info": "🔵"
        }
        
        def _update():
            self.header_status.config(text=f"{colors.get(status_type, '●')} {message}")
        
        self.root.after(0, _update)
    
    def ask_question(self):
        """질문 처리"""
        question = self.question_text.get("1.0", tk.END).strip()
        
        if not question:
            messagebox.showwarning("입력 오류", "질문을 입력해주세요.")
            return
        
        if not self.initialized:
            messagebox.showwarning("시스템 오류", "시스템이 아직 초기화되지 않았습니다.")
            return
        
        self.current_question = question
        
        # 버튼 비활성화
        self.ask_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        
        # 결과 영역 초기화
        self.clear_results()
        
        # 로딩 오버레이 표시
        self.loading_overlay.show("AI가 답변을 생성하고 있습니다...")
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self.process_question, daemon=True)
        thread.start()
    
    def process_question(self):
        """질문 처리 로직"""
        try:
            self.update_header_status("문서 검색 중...", "warning")
            
            use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
            top_k = 3 if use_gpu else 2
            
            results = self.retriever.search(self.current_question, top_k=top_k)
            context_str = build_context(results)
            
            self.root.after(0, lambda: self.display_context(context_str))
            
            self.update_header_status("AI 답변 생성 중...", "warning")
            self.root.after(0, lambda: self.loading_overlay.status_label.config(
                text="답변 생성 중... (15-60초 소요)"
            ))
            
            answer = answer_with_llm(self.current_question, context_str)
            self.current_answer = answer
            
            self.root.after(0, lambda: self.display_answer(answer))
            self.root.after(0, self.loading_overlay.hide)
            
            self.update_header_status("완료!", "success")
            
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, self.loading_overlay.hide)
            self.show_error(f"처리 중 오류 발생: {str(e)}")
            self.update_header_status("처리 실패", "error")
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
    
    def display_context(self, context: str):
        """문맥 표시"""
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.insert("1.0", context)
        self.context_text.config(state=tk.DISABLED)
    
    def display_answer(self, answer: str):
        """답변 표시"""
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", answer)
        self.answer_text.config(state=tk.DISABLED)
    
    def clear_question(self):
        """질문 지우기"""
        self.question_text.delete("1.0", tk.END)
        self.question_text.focus()
    
    def clear_results(self):
        """결과 영역 지우기"""
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.config(state=tk.DISABLED)
        
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.config(state=tk.DISABLED)
    
    def save_answer(self):
        """답변 저장"""
        if not self.current_answer or not self.current_question:
            messagebox.showwarning("저장 오류", "저장할 답변이 없습니다.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_question = "".join(c if c.isalnum() or c in (' ', '_') else '_' 
                                for c in self.current_question[:30])
        filename = f"{safe_question}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"질문: {self.current_question}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"답변:\n{self.current_answer}\n\n")
                f.write("=" * 50 + "\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            messagebox.showinfo("저장 완료", f"답변이 저장되었습니다:\n{filepath}")
            
        except Exception as e:
            self.show_error(f"저장 실패: {str(e)}")
    
    def show_error(self, message: str):
        """에러 메시지 표시"""
        def _show():
            messagebox.showerror("오류", message)
        
        self.root.after(0, _show)


def main():
    """GUI 애플리케이션 실행"""
    root = tk.Tk()
    app = RAGSystemModernGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
