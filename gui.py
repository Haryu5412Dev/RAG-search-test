"""
RAG 통합 프로그램 - GUI 버전
Modern Apple-style UI with Dark/Light Mode

CLI 기능을 유지하면서 GUI로도 사용할 수 있습니다.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import sys
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
    
    # 라이트 모드
    LIGHT = {
        'bg': '#FFFFFF',
        'fg': '#1D1D1F',
        'bg_secondary': '#F5F5F7',
        'border': '#D2D2D7',
        'accent': '#007AFF',
        'accent_hover': '#0051D5',
        'success': '#34C759',
        'warning': '#FF9500',
        'error': '#FF3B30',
        'text_secondary': '#86868B',
        'input_bg': '#FFFFFF',
        'button_bg': '#007AFF',
        'button_fg': '#FFFFFF',
        'card_bg': '#FFFFFF',
        'shadow': '#00000015',
    }
    
    # 다크 모드
    DARK = {
        'bg': '#1C1C1E',
        'fg': '#FFFFFF',
        'bg_secondary': '#2C2C2E',
        'border': '#38383A',
        'accent': '#0A84FF',
        'accent_hover': '#409CFF',
        'success': '#30D158',
        'warning': '#FF9F0A',
        'error': '#FF453A',
        'text_secondary': '#98989D',
        'input_bg': '#2C2C2E',
        'button_bg': '#0A84FF',
        'button_fg': '#FFFFFF',
        'card_bg': '#2C2C2E',
        'shadow': '#00000030',
    }


class ModernButton(tk.Canvas):
    """Apple 스타일 버튼"""
    
    def __init__(self, parent, text, command, width=120, height=36, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, **kwargs)
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.is_hovered = False
        self.is_disabled = False
        
        self.bind('<Button-1>', self._on_click)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        
        self.draw()
    
    def draw(self):
        self.delete('all')
        theme = self.master.current_theme
        
        if self.is_disabled:
            bg = theme['text_secondary']
            fg = theme['bg']
        elif self.is_hovered:
            bg = theme['accent_hover']
            fg = theme['button_fg']
        else:
            bg = theme['button_bg']
            fg = theme['button_fg']
        
        self.config(bg=self.master.current_theme['bg'])
        
        # 둥근 사각형
        self.create_rounded_rect(0, 0, self.width, self.height, 
                                 radius=8, fill=bg, outline='')
        
        # 텍스트
        self.create_text(self.width/2, self.height/2, 
                        text=self.text, fill=fg,
                        font=('SF Pro Display', 11, 'bold'))
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()
    
    def _on_enter(self, event):
        if not self.is_disabled:
            self.is_hovered = True
            self.draw()
    
    def _on_leave(self, event):
        self.is_hovered = False
        self.draw()
    
    def set_state(self, state):
        self.is_disabled = (state == 'disabled')
        self.draw()


class LoadingIndicator(tk.Canvas):
    """Apple 스타일 로딩 인디케이터"""
    
    def __init__(self, parent, size=30):
        super().__init__(parent, width=size, height=size, 
                        highlightthickness=0)
        self.size = size
        self.angle = 0
        self.is_running = False
        self.animation_id = None
        
    def start(self):
        self.is_running = True
        self.animate()
    
    def stop(self):
        self.is_running = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.delete('all')
    
    def animate(self):
        if not self.is_running:
            return
        
        self.delete('all')
        theme = self.master.current_theme
        self.config(bg=theme['bg'])
        
        # 회전하는 원형 로딩 인디케이터
        center = self.size / 2
        radius = self.size / 3
        
        for i in range(8):
            angle = self.angle + (i * 45)
            x = center + radius * tk.math.cos(tk.math.radians(angle))
            y = center + radius * tk.math.sin(tk.math.radians(angle))
            
            opacity = int(255 * (i / 8))
            color = f"#{opacity:02x}{opacity:02x}{opacity:02x}"
            
            self.create_oval(x-2, y-2, x+2, y+2, fill=color, outline='')
        
        self.angle = (self.angle + 10) % 360
        self.animation_id = self.after(50, self.animate)


class LoadingSplash:
    """초기화 중 표시할 로딩 스플래시 스크린"""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("RAG 시스템 초기화")
        self.window.geometry("400x250")
        self.window.resizable(False, False)
        
        # 창을 화면 중앙에 배치
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.window.winfo_screenheight() // 2) - (250 // 2)
        self.window.geometry(f"400x250+{x}+{y}")
        
        # 다른 창 위에 표시
        self.window.attributes('-topmost', True)
        
        # 메인 프레임
        main_frame = ttk.Frame(self.window, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        title_label = ttk.Label(
            main_frame,
            text="RAG 문서 검색 시스템",
            font=("맑은 고딕", 16, "bold")
        )
        title_label.pack(pady=(10, 5))
        
        # 부제목
        subtitle_label = ttk.Label(
            main_frame,
            text="Qwen3-4B 기반 질의응답",
            font=("맑은 고딕", 10)
        )
        subtitle_label.pack(pady=(0, 20))
        
        # 상태 메시지
        self.status_label = ttk.Label(
            main_frame,
            text="초기화 중...",
            font=("맑은 고딕", 10),
            foreground="gray"
        )
        self.status_label.pack(pady=(10, 10))
        
        # 진행 바
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=(10, 20))
        self.progress.start(10)
        
        # 상세 정보
        self.detail_label = ttk.Label(
            main_frame,
            text="",
            font=("맑은 고딕", 8),
            foreground="darkgray"
        )
        self.detail_label.pack(pady=(5, 10))
    
    def update_status(self, message: str, detail: str = ""):
        """상태 메시지 업데이트"""
        self.status_label.config(text=message)
        if detail:
            self.detail_label.config(text=detail)
        self.window.update()
    
    def close(self):
        """로딩 창 닫기"""
        self.progress.stop()
        self.window.destroy()


class RAGSystemGUI:
    """RAG 시스템 GUI 애플리케이션"""

    def __init__(self, root, splash):
        self.root = root
        self.splash = splash
        self.root.title("RAG 문서 검색 시스템")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)
        
        # 초기에는 메인 창 숨기기
        self.root.withdraw()
        
        # 환경 변수 로드
        load_dotenv()
        
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
        
        # 자동 초기화 시작
        self.root.after(500, self.auto_initialize)
    
    def setup_ui(self):
        """UI 레이아웃 구성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=2)
        
        # ===== 상단: 질문 입력 영역 =====
        question_frame = ttk.LabelFrame(main_frame, text="질문 입력", padding="10")
        question_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        question_frame.columnconfigure(0, weight=1)
        
        self.question_text = scrolledtext.ScrolledText(
            question_frame, 
            height=3, 
            wrap=tk.WORD,
            font=("맑은 고딕", 10)
        )
        self.question_text.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 버튼 프레임
        button_frame = ttk.Frame(question_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.ask_button = ttk.Button(
            button_frame, 
            text="질문하기", 
            command=self.ask_question,
            width=15
        )
        self.ask_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(
            button_frame, 
            text="입력 지우기", 
            command=self.clear_question,
            width=15
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(
            button_frame, 
            text="답변 저장", 
            command=self.save_answer,
            width=15,
            state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # ===== 중간: 상태 표시 영역 =====
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(
            status_frame, 
            text="초기화 중...", 
            font=("맑은 고딕", 9),
            foreground="gray"
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(
            status_frame, 
            mode='indeterminate', 
            length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=10)
        
        # ===== 하단: 검색된 문맥 영역 =====
        context_frame = ttk.LabelFrame(main_frame, text="검색된 문맥", padding="10")
        context_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        context_frame.columnconfigure(0, weight=1)
        
        self.context_text = scrolledtext.ScrolledText(
            context_frame, 
            height=6, 
            wrap=tk.WORD,
            font=("맑은 고딕", 9),
            state=tk.DISABLED
        )
        self.context_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # ===== 하단: 답변 출력 영역 =====
        answer_frame = ttk.LabelFrame(main_frame, text="답변", padding="10")
        answer_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        answer_frame.columnconfigure(0, weight=1)
        answer_frame.rowconfigure(0, weight=1)
        
        self.answer_text = scrolledtext.ScrolledText(
            answer_frame, 
            wrap=tk.WORD,
            font=("맑은 고딕", 10),
            state=tk.DISABLED
        )
        self.answer_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ===== 하단 정보 바 =====
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        gpu_status = "GPU 모드" if use_gpu else "CPU 모드"
        
        self.info_label = ttk.Label(
            info_frame, 
            text=f"모델: Qwen3-4B | 실행 모드: {gpu_status}",
            font=("맑은 고딕", 8),
            foreground="gray"
        )
        self.info_label.pack(side=tk.LEFT)
        
        # 메뉴바 추가
        self.setup_menu()
    
    def setup_menu(self):
        """메뉴바 구성"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="답변 저장", command=self.save_answer)
        file_menu.add_command(label="출력 폴더 열기", command=self.open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)
        
        # 설정 메뉴
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="설정", menu=settings_menu)
        settings_menu.add_command(label="시스템 재초기화", command=self.reinitialize)
        
        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="사용 방법", command=self.show_help)
        help_menu.add_command(label="정보", command=self.show_about)
    
    def auto_initialize(self):
        """자동 초기화"""
        thread = threading.Thread(target=self.initialize_system, daemon=True)
        thread.start()
    
    def initialize_system(self):
        """시스템 초기화 (별도 스레드)"""
        try:
            # 스플래시 스크린 업데이트
            if self.splash:
                self.splash.update_status("문서 로딩 중...", "PLM 매뉴얼 확인")
            self.update_status("문서 로딩 중...")
            self.progress_bar.start()
            
            # 문서 확인
            if not self.document_path.exists():
                self.show_error(f"문서를 찾을 수 없습니다: {self.document_path}")
                if self.splash:
                    self.splash.close()
                return
            
            # 청크 로드
            if self.splash:
                self.splash.update_status("문서 청킹 중...", "텍스트 분할 작업")
            self.update_status("문서 청킹 중...")
            self.chunks = load_chunks(str(self.document_path))
            
            if not self.chunks:
                self.show_error("청크를 생성할 수 없습니다.")
                if self.splash:
                    self.splash.close()
                return
            
            # 검색기 초기화
            if self.splash:
                self.splash.update_status("검색 엔진 초기화 중...", f"{len(self.chunks)}개 청크 인덱싱")
            self.update_status("검색 엔진 초기화 중...")
            self.retriever = TfidfRetriever()
            self.retriever.fit(self.chunks)
            
            # 모델 사전 로드
            if self.splash:
                self.splash.update_status("AI 모델 로딩 중...", "Qwen3-4B 모델 (1-2분 소요)")
            self.update_status("AI 모델 로딩 중... (최대 1-2분 소요)")
            preload_model()
            
            self.initialized = True
            self.progress_bar.stop()
            self.update_status(f"준비 완료! (총 {len(self.chunks)}개 청크)", "green")
            
            # 버튼 활성화
            self.root.after(0, lambda: self.ask_button.config(state=tk.NORMAL))
            
            # 스플래시 닫고 메인 창 표시
            if self.splash:
                self.root.after(0, self.splash.close)
                self.root.after(100, self.root.deiconify)
            
        except Exception as e:
            self.progress_bar.stop()
            self.show_error(f"초기화 실패: {str(e)}")
            self.update_status("초기화 실패", "red")
            if self.splash:
                self.splash.close()
                self.root.deiconify()
            
        except Exception as e:
            self.progress_bar.stop()
            self.show_error(f"초기화 실패: {str(e)}")
            self.update_status("초기화 실패", "red")
    
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
        self.ask_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        # 답변 영역 초기화
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.config(state=tk.DISABLED)
        
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.config(state=tk.DISABLED)
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self.process_question, daemon=True)
        thread.start()
    
    def process_question(self):
        """질문 처리 로직 (별도 스레드)"""
        try:
            self.progress_bar.start()
            self.update_status("관련 문서 검색 중...")
            
            # 검색
            use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
            top_k = 3 if use_gpu else 2
            
            results = self.retriever.search(self.current_question, top_k=top_k)
            
            # 문맥 구성
            self.update_status("답변 생성 중...")
            context_str = build_context(results)
            
            # 문맥 표시
            self.root.after(0, lambda: self.display_context(context_str))
            
            # LLM 답변 생성
            answer = answer_with_llm(self.current_question, context_str)
            
            self.current_answer = answer
            
            # 답변 표시
            self.root.after(0, lambda: self.display_answer(answer))
            
            self.progress_bar.stop()
            self.update_status("완료!", "green")
            
            # 버튼 활성화
            self.root.after(0, lambda: self.ask_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_button.config(state=tk.NORMAL))
            
        except Exception as e:
            self.progress_bar.stop()
            self.show_error(f"처리 중 오류 발생: {str(e)}")
            self.update_status("처리 실패", "red")
            self.root.after(0, lambda: self.ask_button.config(state=tk.NORMAL))
    
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
        """질문 입력 지우기"""
        self.question_text.delete("1.0", tk.END)
        self.question_text.focus()
    
    def save_answer(self):
        """답변 저장"""
        if not self.current_answer or not self.current_question:
            messagebox.showwarning("저장 오류", "저장할 답변이 없습니다.")
            return
        
        # 파일명 생성
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
    
    def open_output_folder(self):
        """출력 폴더 열기"""
        try:
            os.startfile(self.output_dir)
        except Exception as e:
            self.show_error(f"폴더 열기 실패: {str(e)}")
    
    def reinitialize(self):
        """시스템 재초기화"""
        if messagebox.askyesno("재초기화", "시스템을 재초기화하시겠습니까?"):
            self.initialized = False
            self.retriever = None
            self.chunks = []
            self.ask_button.config(state=tk.DISABLED)
            self.auto_initialize()
    
    def show_help(self):
        """사용 방법 표시"""
        help_text = """
RAG 문서 검색 시스템 사용 방법

1. 질문 입력창에 질문을 입력합니다.
2. '질문하기' 버튼을 클릭합니다.
3. 시스템이 관련 문서를 검색하고 답변을 생성합니다.
4. 답변을 확인한 후 '답변 저장' 버튼으로 저장할 수 있습니다.

Tips:
- 구체적이고 명확한 질문을 입력하세요.
- 답변은 output 폴더에 자동 저장됩니다.
- GPU 모드가 CPU 모드보다 빠릅니다.
        """
        messagebox.showinfo("사용 방법", help_text.strip())
    
    def show_about(self):
        """정보 표시"""
        about_text = """
RAG 문서 검색 시스템
버전: 1.0.0

Qwen3-4B 모델 기반
문서 기반 질의응답 시스템

개발: 2026
        """
        messagebox.showinfo("정보", about_text.strip())
    
    def update_status(self, message: str, color: str = "gray"):
        """상태 메시지 업데이트"""
        def _update():
            self.status_label.config(text=message, foreground=color)
        
        self.root.after(0, _update)
    
    def show_error(self, message: str):
        """에러 메시지 표시"""
        def _show():
            messagebox.showerror("오류", message)
        
        self.root.after(0, _show)


def main():
    """GUI 애플리케이션 실행"""
    # 메인 루트 창 생성 (숨김)
    root = tk.Tk()
    root.withdraw()
    
    # 스플래시 스크린 표시
    splash = LoadingSplash(root)
    splash.update_status("시스템 준비 중...", "UI 초기화")
    
    # 메인 앱 생성
    app = RAGSystemGUI(root, splash)
    
    # 이벤트 루프 시작
    root.mainloop()


if __name__ == "__main__":
    main()
