# 세이프펫 체커 - 해외직구 반려용품 유해성분 검사 앱 (스트림릿)
# 실행: 터미널에서  streamlit run app.py
# 파일 구성: app.py / ingredients.csv(유해성분 표) / color_levels.csv(색 단계표) / requirements.txt

import re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="세이프펫 체커", page_icon="🐾", layout="centered")

# ---------------------------------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------------------------------
@st.cache_data
def load_ingredients():
    return pd.read_csv("ingredients.csv")

@st.cache_data
def load_levels():
    return pd.read_csv("color_levels.csv")

# 제품 종류별 판정 기준 (μg/g). (주의 시작값, 위험 시작값)
# 구강스프레이·물티슈: 입과 점막에 바로 닿으므로 20 μg/g(물휴지 기준)을 넘으면 위험
# 샴푸: 씻어내는 제품이지만 핥아서 삼킬 수 있어 주의 구간을 넓게 잡음 (화장품 검출 한도 2000 μg/g)
THRESHOLD = {
    "구강스프레이": (10, 20),
    "물티슈": (10, 20),
    "샴푸": (20, 2000),
}

# ---------------------------------------------------------------
# 2. 성분 검색 기능
# ---------------------------------------------------------------
def normalize(text: str) -> str:
    """대소문자, 공백, 하이픈, 점, 괄호를 없애서 표기가 달라도 비교되게 만든다."""
    return re.sub(r"[\s\-\.\(\)\[\]_/]", "", str(text)).lower()

def split_ingredients(text: str):
    """전성분 문자열을 쉼표·줄바꿈·세미콜론·가운뎃점 기준으로 나눈다.
    '1,3-diol'처럼 숫자 사이의 쉼표는 성분 이름의 일부이므로 나누지 않는다."""
    protected = re.sub(r"(?<=\d),(?=\d)", "§", text)
    parts = re.split(r"[,\n;·|]", protected)
    return [p.strip().replace("§", ",") for p in parts if p.strip()]

def find_hazards(text: str, db: pd.DataFrame):
    tokens = split_ingredients(text)
    hits = []
    for tok in tokens:
        n_tok = normalize(tok)
        for _, row in db.iterrows():
            names = [row["성분명"]] + str(row["다른이름"]).split(";")
            for name in names:
                n_name = normalize(name)
                # MIT, MCI처럼 짧은 약어는 토큰 전체가 같을 때만 인정 (Vitamin 안의 'mi' 같은 오검출 방지)
                matched = (n_tok == n_name) if len(n_name) <= 4 else (n_name and n_name in n_tok)
                if matched:
                    hits.append({"입력한 성분": tok, **row.to_dict()})
                    break
            else:
                continue
            break
    return pd.DataFrame(hits)

def grade_from_hits(hits: pd.DataFrame) -> str:
    if hits.empty:
        return "안전"
    if (hits["등급"] == "위험").any():
        return "위험"
    return "주의"

# ---------------------------------------------------------------
# 3. 색 분석 기능
# ---------------------------------------------------------------
def average_rgb(img: Image.Image, box_frac: float = 0.2):
    """사진 가운데 정사각형 영역(기본 20%)의 평균 RGB를 구한다."""
    arr = np.asarray(img.convert("RGB")).astype(float)
    h, w, _ = arr.shape
    cy, cx = h // 2, w // 2
    dh, dw = max(1, int(h * box_frac / 2)), max(1, int(w * box_frac / 2))
    region = arr[cy - dh:cy + dh, cx - dw:cx + dw]
    return region.reshape(-1, 3).mean(axis=0)

def white_balance(rgb, white_rgb):
    """흰색 기준 사진의 RGB가 (255,255,255)가 되도록 배율을 맞춰 조명 차이를 줄인다."""
    gain = 255.0 / np.maximum(np.asarray(white_rgb, dtype=float), 1.0)
    return np.clip(np.asarray(rgb) * gain, 0, 255)

def nearest_level(rgb, levels: pd.DataFrame):
    """색 단계표에서 RGB 거리가 가장 가까운 농도를 고른다."""
    ref = levels[["R", "G", "B"]].to_numpy(dtype=float)
    dist = np.sqrt(((ref - np.asarray(rgb)) ** 2).sum(axis=1))
    i = int(np.argmin(dist))
    return levels.iloc[i], float(dist[i])

def grade_from_conc(conc_ugg: float, product: str) -> str:
    caution, danger = THRESHOLD[product]
    if conc_ugg >= danger:
        return "위험"
    if conc_ugg >= caution:
        return "주의"
    return "안전"

def show_grade(grade: str, message: str):
    if grade == "위험":
        st.error(f"판정: 위험  -  {message}")
    elif grade == "주의":
        st.warning(f"판정: 주의  -  {message}")
    else:
        st.success(f"판정: 안전  -  {message}")

# ---------------------------------------------------------------
# 4. 화면
# ---------------------------------------------------------------
st.title("🐾 세이프펫 체커")
st.caption("해외직구 반려용품, 성분표와 간이 시험지로 집에서 1차 확인하기")

tab1, tab2, tab3, tab4 = st.tabs(["성분 검색", "색 분석", "기준 색 등록", "판정 기준 안내"])

# ---- 탭 1: 성분 검색 ----
with tab1:
    st.subheader("전성분 붙여 넣기")
    st.write("제품 라벨이나 판매 페이지의 전성분을 그대로 복사해 붙여 넣으면 위험 성분을 찾아 줍니다.")
    text = st.text_area("전성분", height=150,
                        placeholder="예) Water, Sodium Laureth Sulfate, Cocamidopropyl Betaine, DMDM Hydantoin, Fragrance ...")
    if st.button("검사하기", key="btn_ingredient"):
        db = load_ingredients()
        if not text.strip():
            st.info("전성분을 먼저 입력해 주세요.")
        else:
            hits = find_hazards(text, db)
            grade = grade_from_hits(hits)
            if grade == "안전":
                show_grade(grade, "등록된 위험 성분이 없습니다. 다만 표시되지 않은 성분은 확인할 수 없습니다.")
            else:
                show_grade(grade, f"위험 성분 {len(hits)}개가 있습니다.")
                for _, r in hits.iterrows():
                    with st.expander(f"{r['등급']}  |  {r['성분명']}  (입력: {r['입력한 성분']})"):
                        st.write(f"분류: {r['분류']}")
                        st.write(f"국내 기준: {r['국내기준']}")
                        st.write(f"왜 위험한가: {r['설명']}")
                if (hits["분류"] == "방출형 방부제").any():
                    st.info("방출형 방부제는 라벨에 포름알데히드가 없어도 물속에서 포름알데히드를 만들어 냅니다. "
                            "간이 시험지로 실제 농도를 확인해 보세요. (색 분석 탭)")
            st.caption("판정은 사람용 화장품 기준을 빌려 쓴 것입니다. 개·고양이는 피부가 얇고 몸을 핥기 때문에 실제 위험은 더 클 수 있습니다.")

# ---- 탭 2: 색 분석 ----
with tab2:
    st.subheader("시험지 사진으로 농도 읽기")
    st.write("흰 배경 위에서 플래시를 끄고, 시험지(또는 팩테스트 튜브)가 사진 가운데 오도록 찍어 올려 주세요.")
    product = st.selectbox("제품 종류", list(THRESHOLD.keys()))
    dilution = st.number_input("희석 배수 (제품 1 g을 물 99 mL에 풀었으면 100)", min_value=1, value=100, step=1)
    photo = st.file_uploader("시험지 사진", type=["jpg", "jpeg", "png"], key="sample")
    with st.expander("조명 보정 (선택): 같은 조건에서 찍은 흰 종이 사진"):
        white = st.file_uploader("흰 종이 사진", type=["jpg", "jpeg", "png"], key="white")

    if photo is not None:
        img = Image.open(photo)
        st.image(img, caption="올린 사진 (가운데 20% 영역을 분석)", width=300)
        rgb = average_rgb(img)
        if white is not None:
            rgb = white_balance(rgb, average_rgb(Image.open(white)))
        levels = load_levels()
        level, dist = nearest_level(rgb, levels)
        conc_mgL = float(level["농도_mgL"])
        conc_ugg = conc_mgL * dilution  # 묽힌 용액 1 L ≈ 1000 g 이므로 mg/L ≈ μg/g, 여기에 희석 배수를 곱한다
        grade = grade_from_conc(conc_ugg, product)

        c1, c2, c3 = st.columns(3)
        c1.metric("평균 RGB", f"{int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}")
        c2.metric("가장 가까운 단계", f"{conc_mgL} mg/L")
        c3.metric("제품 속 농도(추정)", f"{conc_ugg:,.0f} μg/g")
        show_grade(grade, f"{product} 기준 (주의 {THRESHOLD[product][0]} μg/g, 위험 {THRESHOLD[product][1]} μg/g 이상)")
        if dist > 60:
            st.warning("색 단계표와 색이 많이 다릅니다. 조명 보정을 하거나 다시 촬영해 보세요.")
        if "예시값" in str(level.get("비고", "")):
            st.info("아직 예시 색 단계표를 쓰고 있습니다. 검출 실험 후 '기준 색 등록' 탭에서 실제 값으로 바꿔 주세요.")

# ---- 탭 3: 기준 색 등록 도우미 ----
with tab3:
    st.subheader("색 단계표 만들기 도우미")
    st.write("표준 용액(또는 키트의 색대조표) 사진을 한 장씩 올리면 평균 RGB를 알려 줍니다. "
             "그 값을 color_levels.csv에 적어 넣으면 색 분석 탭이 그 표를 씁니다.")
    ref_photo = st.file_uploader("기준 사진", type=["jpg", "jpeg", "png"], key="ref")
    if ref_photo is not None:
        rgb = average_rgb(Image.open(ref_photo))
        st.code(f"R={int(rgb[0])}, G={int(rgb[1])}, B={int(rgb[2])}")
    st.dataframe(load_levels())

# ---- 탭 4: 판정 기준 안내 ----
with tab4:
    st.subheader("판정 기준")
    st.markdown("""
- **구강스프레이·물티슈**: 입과 점막에 바로 닿는 제품이라 물휴지 기준 **20 μg/g** 이상이면 위험, 10 μg/g 이상이면 주의
- **샴푸**: 씻어내는 제품이지만 핥아서 삼킬 수 있어 **20 μg/g 이상 주의**, 화장품 검출 한도 **2000 μg/g 이상 위험**
- **성분 검색**: 포름알데히드·CMIT/MIT는 위험, 방출형 방부제·벤조산은 주의

이 기준은 사람용 화장품 안전기준을 빌려 쓴 것입니다. 국내에는 반려동물 전용 유해물질 기준이 아직 없습니다.
개·고양이는 표피가 얇고 피부 pH가 중성에 가까우며 몸을 핥기 때문에 같은 농도라도 사람보다 위험할 수 있습니다.
간이 시험은 정밀 검사를 대신하지 못합니다. 피부 이상이 계속되면 동물병원에 가세요.
    """)
    st.caption("자료: 한국소비자원 해외직구 반려동물 용품 안전실태조사(2024.9), 화장품 안전기준 등에 관한 규정")
