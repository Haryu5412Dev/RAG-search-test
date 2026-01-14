"""
RAG 문서 검색 시스템 - Modern GUI
깔끔하고 세련된 인터페이스
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from util.chunking import load_chunks
from util.generation import answer_with_llm, build_context, preload_model
from util.retrieval import TfidfRetriever


class Colors:
    """컬러 테마 (Apple 느낌)"""
    LIGHT = {
        'primary_bg': '#F5F5F7',
        'secondary_bg': '#FFFFFF',
        'card_bg': '#FFFFFF',
        'text': '#1C1C1E',
        'text_muted': '#6E6E73',
        'border': '#D2D2D7',
        'accent': '#007AFF',
        'accent_hover': '#0A6DFF',
        'success': '#34C759',
        'danger': '#FF3B30',
        'input_bg': '#F9F9FB',
    }

    DARK = {
        'primary_bg': '#1C1C1E',
        'secondary_bg': '#2C2C2E',
        'card_bg': '#2C2C2E',
        'text': '#F2F2F7',
        'text_muted': '#A1A1AA',
        'border': '#3A3A3C',
        'accent': '#0A84FF',
        'accent_hover': '#3A9BFF',
        'success': '#30D158',
        'danger': '#FF453A',
        'input_bg': '#1F1F22',
    }


class LoadingDialog:
    """전체 창을 덮는 로딩 오버레이"""

    def __init__(self, parent):
        self.parent = parent
        self.overlay: Optional[tk.Frame] = None
        self.status_label: Optional[tk.Label] = None
        self.is_showing = False

    def show(self, theme, message="AI 답변 생성 중..."):
        if self.is_showing:
            return
        self.is_showing = True

        self.overlay = tk.Frame(self.parent, bg=theme['secondary_bg'])
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        inner = tk.Frame(self.overlay, bg=theme['secondary_bg'], padx=26, pady=22)
        inner.place(relx=0.5, rely=0.5, anchor='center')
        inner.configure(highlightthickness=1, highlightbackground=theme['border'])

        tk.Label(
            inner,
            text=message,
            font=("SF Pro Display", 13, "bold"),
            bg=theme['secondary_bg'],
            fg=theme['text']
        ).pack(pady=(0, 10))

        self.status_label = tk.Label(
            inner,
            text="문서 검색 중",
            font=("SF Pro Text", 10),
            bg=theme['secondary_bg'],
            fg=theme['text_muted']
        )
        self.status_label.pack()

    def update_status(self, text):
        if self.status_label:
            self.status_label.config(text=text)
            self.status_label.update_idletasks()

    def close(self):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        self.is_showing = False


class RAGApp:
    """RAG 시스템 애플리케이션"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RAG 문서 검색")
        self.root.geometry("1100x800")
        self.root.minsize(900, 700)
        
        load_dotenv()
        
        # 시스템 변수
        self.document_path = Path("data/PLM_training_manual_clean.txt")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.retriever: Optional[TfidfRetriever] = None
        self.chunks = []
        self.initialized = False
        self.current_question = ""
        self.current_answer = ""
        
        # 테마
        self.is_dark = False
        self.theme = Colors.LIGHT
        self.primary_frames = []
        self.secondary_frames = []
        self.card_frames = []
        self.primary_labels = []
        
        # UI 구성
        self.create_ui()
        
        # 초기화
        self.root.after(200, self.initialize)
    
    def create_ui(self):
        """UI 생성"""
        # 메인 컨테이너
        self.main_frame = tk.Frame(self.root, bg=self.theme['primary_bg'])
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.primary_frames.append(self.main_frame)
        
        # 헤더
        self.create_header()
        
        # 컨텐츠
        self.content_frame = tk.Frame(self.main_frame, bg=self.theme['primary_bg'])
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 30))
        self.primary_frames.append(self.content_frame)
        
        # 질문 섹션
        self.create_question_section(self.content_frame)
        
        # 버튼
        self.create_buttons(self.content_frame)
        
        # 결과 섹션
        self.create_results_section(self.content_frame)
        
        # 푸터
        self.create_footer()
        
        self.apply_theme()
    
    def create_header(self):
        """헤더 생성"""
        self.header = tk.Frame(self.main_frame, bg=self.theme['secondary_bg'], height=100)
        self.header.pack(fill=tk.X)
        self.header.pack_propagate(False)
        self.secondary_frames.append(self.header)
        
        self.header_inner = tk.Frame(self.header, bg=self.theme['secondary_bg'])
        self.header_inner.pack(fill=tk.BOTH, padx=40, pady=20)
        self.secondary_frames.append(self.header_inner)
        
        # 왼쪽
        self.header_left = tk.Frame(self.header_inner, bg=self.theme['secondary_bg'])
        self.header_left.pack(side=tk.LEFT, fill=tk.Y)
        self.secondary_frames.append(self.header_left)
        
        self.title_label = tk.Label(
            self.header_left,
            text="RAG 문서 검색",
            font=("SF Pro Display", 22, 'bold'),
            bg=self.theme['secondary_bg'],
            fg=self.theme['text']
        )
        self.title_label.pack(anchor=tk.W)
        
        self.subtitle_label = tk.Label(
            self.header_left,
            text="Qwen3-4B 기반 질의응답 시스템",
            font=("SF Pro Text", 10),
            bg=self.theme['secondary_bg'],
            fg=self.theme['text_muted']
        )
        self.subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        # 오른쪽
        self.header_right = tk.Frame(self.header_inner, bg=self.theme['secondary_bg'])
        self.header_right.pack(side=tk.RIGHT, fill=tk.Y)
        self.secondary_frames.append(self.header_right)
        
        self.status_label = tk.Label(
            self.header_right,
            text="초기화 중...",
            font=("SF Pro Text", 10),
            bg=self.theme['secondary_bg'],
            fg=self.theme['text_muted']
        )
        self.status_label.pack(anchor=tk.E)
        
        self.theme_btn = tk.Button(
            self.header_right,
            text="다크 모드",
            font=("SF Pro Text", 10),
            command=self.toggle_theme,
            relief=tk.FLAT,
            cursor='hand2',
            padx=16,
            pady=8,
            bg=self.theme['primary_bg'],
            fg=self.theme['text']
        )
        self.theme_btn.pack(anchor=tk.E, pady=(8, 0))
    
    def create_question_section(self, parent):
        """질문 입력 섹션"""
        self.question_section = tk.Frame(parent, bg=self.theme['primary_bg'])
        self.question_section.pack(fill=tk.X, pady=(20, 0))
        self.primary_frames.append(self.question_section)
        
        self.question_label = tk.Label(
            self.question_section,
            text="질문을 입력하세요",
            font=("SF Pro Display", 14, 'bold'),
            bg=self.theme['primary_bg'],
            fg=self.theme['text']
        )
        self.question_label.pack(anchor=tk.W, pady=(0, 12))
        self.primary_labels.append(self.question_label)
        
        # 카드
        self.question_card = tk.Frame(self.question_section, bg=self.theme['card_bg'], relief=tk.FLAT)
        self.question_card.pack(fill=tk.X)
        self.card_frames.append(self.question_card)
        
        self.question_text = scrolledtext.ScrolledText(
            self.question_card,
            height=4,
            font=("SF Pro Text", 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=16,
            pady=16,
            bg=self.theme['input_bg'],
            fg=self.theme['text'],
            insertbackground=self.theme['text'],
            selectbackground=self.theme['accent'],
            selectforeground='#FFFFFF',
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.theme['border']
        )
        self.question_text.pack(fill=tk.BOTH, padx=2, pady=2)
    
    def create_buttons(self, parent):
        """버튼 생성"""
        self.buttons_frame = tk.Frame(parent, bg=self.theme['primary_bg'])
        self.buttons_frame.pack(fill=tk.X, pady=20)
        self.primary_frames.append(self.buttons_frame)
        
        self.ask_btn = tk.Button(
            self.buttons_frame,
            text="질문하기",
            font=("SF Pro Text", 11, 'bold'),
            command=self.ask_question,
            relief=tk.FLAT,
            cursor='hand2',
            padx=35,
            pady=14,
            bg=self.theme['accent'],
            fg='#FFFFFF',
            activebackground=self.theme['accent_hover'],
            activeforeground='#FFFFFF',
            state=tk.DISABLED
        )
        self.ask_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = tk.Button(
            self.buttons_frame,
            text="초기화",
            font=("SF Pro Text", 10),
            command=self.clear_all,
            relief=tk.FLAT,
            cursor='hand2',
            padx=25,
            pady=14,
            bg=self.theme['secondary_bg'],
            fg=self.theme['text'],
            activebackground=self.theme['border']
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_btn = tk.Button(
            self.buttons_frame,
            text="답변 저장",
            font=("SF Pro Text", 10),
            command=self.save_answer,
            relief=tk.FLAT,
            cursor='hand2',
            padx=25,
            pady=14,
            bg=self.theme['secondary_bg'],
            fg=self.theme['text'],
            activebackground=self.theme['border'],
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT)
    
    def create_results_section(self, parent):
        """결과 섹션"""
        self.results_section = tk.Frame(parent, bg=self.theme['primary_bg'])
        self.results_section.pack(fill=tk.BOTH, expand=True)
        self.primary_frames.append(self.results_section)
        
        # 검색 문맥
        self.context_label = tk.Label(
            self.results_section,
            text="검색된 문맥",
            font=("SF Pro Display", 13, 'bold'),
            bg=self.theme['primary_bg'],
            fg=self.theme['text']
        )
        self.context_label.pack(anchor=tk.W, pady=(0, 12))
        self.primary_labels.append(self.context_label)
        
        self.context_card = tk.Frame(self.results_section, bg=self.theme['card_bg'])
        self.context_card.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        self.card_frames.append(self.context_card)
        
        self.context_text = scrolledtext.ScrolledText(
            self.context_card,
            height=7,
            font=("SF Pro Text", 10),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=16,
            pady=16,
            bg=self.theme['input_bg'],
            fg=self.theme['text_muted'],
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.theme['border']
        )
        self.context_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # 답변
        self.answer_label = tk.Label(
            self.results_section,
            text="AI 답변",
            font=("SF Pro Display", 13, 'bold'),
            bg=self.theme['primary_bg'],
            fg=self.theme['text']
        )
        self.answer_label.pack(anchor=tk.W, pady=(0, 12))
        self.primary_labels.append(self.answer_label)
        
        self.answer_card = tk.Frame(self.results_section, bg=self.theme['card_bg'])
        self.answer_card.pack(fill=tk.BOTH, expand=True)
        self.card_frames.append(self.answer_card)
        
        self.answer_text = scrolledtext.ScrolledText(
            self.answer_card,
            height=10,
            font=("SF Pro Text", 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=16,
            pady=16,
            bg=self.theme['input_bg'],
            fg=self.theme['text'],
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.theme['border']
        )
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def create_footer(self):
        """푸터"""
        self.footer = tk.Frame(self.main_frame, bg=self.theme['secondary_bg'], height=50)
        self.footer.pack(fill=tk.X)
        self.footer.pack_propagate(False)
        self.secondary_frames.append(self.footer)
        
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        mode_text = "GPU 모드" if use_gpu else "CPU 모드"
        
        self.footer_label = tk.Label(
            self.footer,
            text=f"{mode_text} | Qwen3-4B",
            font=("SF Pro Text", 9),
            bg=self.theme['secondary_bg'],
            fg=self.theme['text_muted']
        )
        self.footer_label.pack(side=tk.LEFT, padx=40)
    
    def toggle_theme(self):
        """테마 전환"""
        self.is_dark = not self.is_dark
        self.theme = Colors.DARK if self.is_dark else Colors.LIGHT
        self.theme_btn.config(text="라이트 모드" if self.is_dark else "다크 모드")
        self.apply_theme()
    
    def apply_theme(self):
        """테마 적용"""
        # 루트
        self.root.config(bg=self.theme['primary_bg'])
        self.main_frame.config(bg=self.theme['primary_bg'])
        for frame in self.primary_frames:
            frame.config(bg=self.theme['primary_bg'])
        for frame in self.secondary_frames:
            frame.config(bg=self.theme['secondary_bg'])
        for frame in self.card_frames:
            frame.config(
                bg=self.theme['card_bg'],
                highlightthickness=1,
                highlightbackground=self.theme['border'],
                highlightcolor=self.theme['border']
            )
        
        # 헤더/푸터
        for widget in [self.title_label, self.subtitle_label, self.status_label]:
            widget.config(bg=self.theme['secondary_bg'], fg=self.theme['text'])
        
        self.subtitle_label.config(fg=self.theme['text_muted'])
        self.status_label.config(fg=self.theme['text_muted'])
        for lbl in self.primary_labels:
            lbl.config(bg=self.theme['primary_bg'], fg=self.theme['text'])
        
        self.theme_btn.config(
            bg=self.theme['primary_bg'],
            fg=self.theme['text'],
            activebackground=self.theme['border'],
            activeforeground=self.theme['text']
        )
        
        self.footer_label.config(
            bg=self.theme['secondary_bg'],
            fg=self.theme['text_muted']
        )
        
        # 버튼
        self.ask_btn.config(
            bg=self.theme['accent'],
            activebackground=self.theme['accent_hover'],
            fg='#FFFFFF',
            activeforeground='#FFFFFF'
        )
        
        for btn in [self.clear_btn, self.save_btn]:
            btn.config(
                bg=self.theme['secondary_bg'],
                fg=self.theme['text'],
                activebackground=self.theme['border'],
                activeforeground=self.theme['text']
            )
        
        # 텍스트
        for text in [self.question_text, self.context_text, self.answer_text]:
            text.config(
                bg=self.theme['input_bg'],
                fg=self.theme['text'],
                insertbackground=self.theme['text'],
                selectbackground=self.theme['accent'],
                highlightbackground=self.theme['border'],
                highlightcolor=self.theme['accent']
            )
        
        self.context_text.config(fg=self.theme['text_muted'])
    
    def initialize(self):
        """초기화"""
        thread = threading.Thread(target=self._init_system, daemon=True)
        thread.start()
    
    def _init_system(self):
        """시스템 초기화"""
        try:
            self.update_status("문서 로딩 중...")
            
            if not self.document_path.exists():
                self.show_error(f"문서를 찾을 수 없습니다: {self.document_path}")
                return
            
            self.update_status("문서 청킹 중...")
            self.chunks = load_chunks(str(self.document_path))
            
            if not self.chunks:
                self.show_error("청크 생성 실패")
                return
            
            self.update_status("검색 엔진 초기화...")
            self.retriever = TfidfRetriever()
            self.retriever.fit(self.chunks)
            
            self.update_status("AI 모델 로딩 중...")
            preload_model()
            
            self.initialized = True
            self.update_status(f"준비 완료 ({len(self.chunks)}개 청크)")
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.show_error(f"초기화 실패: {str(e)}")
            self.update_status("초기화 실패")
    
    def update_status(self, text):
        """상태 업데이트"""
        self.root.after(0, lambda: self.status_label.config(text=text))
    
    def ask_question(self):
        """질문 처리"""
        question = self.question_text.get("1.0", tk.END).strip()
        
        if not question:
            messagebox.showwarning("입력 오류", "질문을 입력해주세요.")
            return
        
        if not self.initialized:
            messagebox.showwarning("오류", "시스템이 초기화되지 않았습니다.")
            return
        
        self.current_question = question
        self.ask_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        
        # 결과 초기화
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.config(state=tk.DISABLED)
        
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.config(state=tk.DISABLED)
        
        # 로딩 다이얼로그
        self.loading = LoadingDialog(self.root)
        self.root.after(0, lambda: self.loading.show(self.theme))
        
        thread = threading.Thread(target=self._process_question, daemon=True)
        thread.start()
    
    def _process_question(self):
        """질문 처리"""
        try:
            self.update_status("문서 검색 중...")
            self.root.after(0, lambda: self.loading.update_status("문서 검색 중..."))
            
            use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
            top_k = 3 if use_gpu else 2
            
            results = self.retriever.search(self.current_question, top_k=top_k)
            context_str = build_context(results)
            
            self.root.after(0, lambda: self.display_context(context_str))
            
            self.update_status("AI 답변 생성 중...")
            self.root.after(0, lambda: self.loading.update_status("답변 생성 중 (15-60초 소요)"))
            
            answer = answer_with_llm(self.current_question, context_str)
            self.current_answer = answer
            
            self.root.after(0, lambda: self.display_answer(answer))
            self.root.after(0, self.loading.close)
            
            self.update_status("완료")
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, self.loading.close)
            self.show_error(f"오류: {str(e)}")
            self.update_status("처리 실패")
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
    
    def display_context(self, text):
        """문맥 표시"""
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.insert("1.0", text)
        self.context_text.config(state=tk.DISABLED)
    
    def display_answer(self, text):
        """답변 표시"""
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", text)
        self.answer_text.config(state=tk.DISABLED)
    
    def clear_all(self):
        """초기화"""
        self.question_text.delete("1.0", tk.END)
        
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.config(state=tk.DISABLED)
        
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.config(state=tk.DISABLED)
        
        self.current_question = ""
        self.current_answer = ""
        self.question_text.focus()
    
    def save_answer(self):
        """답변 저장"""
        if not self.current_answer:
            messagebox.showwarning("오류", "저장할 답변이 없습니다.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_q = "".join(c if c.isalnum() or c == ' ' else '_' 
                        for c in self.current_question[:30])
        filename = f"{safe_q}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"질문: {self.current_question}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"답변:\n{self.current_answer}\n\n")
                f.write("=" * 60 + "\n")
                f.write(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            messagebox.showinfo("완료", f"저장됨:\n{filepath}")
        except Exception as e:
            self.show_error(f"저장 실패: {str(e)}")
    
    def show_error(self, msg):
        """에러 표시"""
        self.root.after(0, lambda: messagebox.showerror("오류", msg))


def main():
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
