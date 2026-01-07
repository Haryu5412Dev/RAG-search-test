from __future__ import annotations

import sys

import build_chunks
import eval_search
import search
import verify_chunks


def _print_menu() -> None:
    print("\n=== 메뉴 ===")
    print("1) 청크 생성 (build_chunks)")
    print("2) 청크 샘플 검증 (verify_chunks)")
    print("3) 검색기 실행 (search)")
    print("4) 평가/답변 생성 (eval_search)")
    print("0) 종료")


def _run(choice: str) -> bool:
    c = choice.strip()
    if c == "1":
        build_chunks.main()
        return True
    if c == "2":
        verify_chunks.main()
        return True
    if c == "3":
        search.main()
        return True
    if c == "4":
        eval_search.main()
        return True
    if c in {"0", "q", "quit", "exit"}:
        return False

    print("잘못된 입력입니다. 1/2/3/4 또는 0을 입력하세요.")
    return True


def main() -> None:
    while True:
        _print_menu()
        choice = input("선택 > ")
        try:
            cont = _run(choice)
        except KeyboardInterrupt:
            print("\n중단되었습니다.")
            return
        except Exception as e:
            print(f"실행 중 오류: {type(e).__name__}: {e}")
            cont = True

        if not cont:
            return


if __name__ == "__main__":
    # Windows CMD에서 한글 출력 깨짐을 최소화
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
