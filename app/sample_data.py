"""In-memory sample data for telecom callbot demo."""

PLANS = [
    {
        "id": "VP_ECO_2GB",
        "name": "Eco 2GB",
        "monthly_price_vnd": 90000,
        "data_gb_per_day": 2,
        "call_minutes_onnet": 100,
        "is_5g": False,
        "segments": ["pho_thong", "nguoi_lon_tuoi"],
    },
    {
        "id": "VP_MAX_6GB",
        "name": "Max 6GB",
        "monthly_price_vnd": 150000,
        "data_gb_per_day": 6,
        "call_minutes_onnet": 300,
        "is_5g": True,
        "segments": ["sinh_vien", "pho_thong", "lam_viec_online"],
    },
    {
        "id": "VP_FAMILY_4GB",
        "name": "Family 4GB",
        "monthly_price_vnd": 220000,
        "data_gb_per_day": 4,
        "call_minutes_onnet": 1000,
        "is_5g": True,
        "segments": ["gia_dinh", "kinh_doanh_nho"],
    },
    {
        "id": "VP_PRO_10GB",
        "name": "Pro 10GB",
        "monthly_price_vnd": 280000,
        "data_gb_per_day": 10,
        "call_minutes_onnet": 500,
        "is_5g": True,
        "segments": ["lam_viec_online", "streamer", "kinh_doanh"],
    },
]

SIM_OFFERS = [
    {
        "id": "SIM_VIP_01",
        "name": "SIM So Dep Than Tai",
        "sim_price_vnd": 199000,
        "bundle": "Tang 3 thang goi VP_MAX_6GB",
        "target_segments": ["kinh_doanh", "pho_thong"],
    },
    {
        "id": "SIM_STUDENT_02",
        "name": "SIM Sinh Vien",
        "sim_price_vnd": 49000,
        "bundle": "Mien phi 2GB/ngay trong 30 ngay dau",
        "target_segments": ["sinh_vien"],
    },
    {
        "id": "SIM_FAMILY_03",
        "name": "SIM Gia Dinh",
        "sim_price_vnd": 99000,
        "bundle": "Mien phi 500 phut noi mang trong 2 thang",
        "target_segments": ["gia_dinh"],
    },
]

SUBSCRIBERS = {
    "0987000001": {
        "phone": "0987000001",
        "name": "Nguyen Van A",
        "current_plan_id": "VP_ECO_2GB",
        "avg_data_usage_gb_per_day": 3.1,
        "avg_call_minutes_per_month": 80,
    },
    "0987000002": {
        "phone": "0987000002",
        "name": "Tran Thi B",
        "current_plan_id": "VP_MAX_6GB",
        "avg_data_usage_gb_per_day": 4.5,
        "avg_call_minutes_per_month": 140,
    },
    "0987000003": {
        "phone": "0987000003",
        "name": "Le Minh C",
        "current_plan_id": "VP_FAMILY_4GB",
        "avg_data_usage_gb_per_day": 2.0,
        "avg_call_minutes_per_month": 620,
    },
}

LEADS = []
CALLBACKS = []
