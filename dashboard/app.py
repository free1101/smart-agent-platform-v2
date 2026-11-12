"""Streamlit 知识库控制台入口。

启动:
    streamlit run dashboard/app.py
    (或 python scripts/run_dashboard.py)

六个页面与主服务解耦:只读数据库 + 调用管理端函数,不依赖 uvicorn 是否在跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

# 项目根必须排在 dashboard/ 前面:否则 `import app.*` 会命中同名的 dashboard/app.py,
# 让 common 变成"部分初始化模块",表现为控制台整页 ImportError(循环导入)。
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# dashboard 目录放末尾即可满足 `from common import ...`,且不会遮蔽顶层 app 包
if str(DASHBOARD_DIR) in sys.path:
    sys.path.remove(str(DASHBOARD_DIR))
sys.path.append(str(DASHBOARD_DIR))

import streamlit as st  # noqa: E402

from common import COLORS, setup_page, tenants  # noqa: E402

setup_page("知识库控制台")

with st.sidebar:
    st.markdown(
        f"<div style='font-size:20px;font-weight:700;color:{COLORS['text']};"
        "letter-spacing:.5px;margin-bottom:2px'>知识库控制台</div>"
        f"<div style='color:{COLORS['muted']};font-size:12px;margin-bottom:18px'>"
        "Smart Agent Platform</div>",
        unsafe_allow_html=True,
    )

    tenant_list = tenants()
    if tenant_list:
        options = {f"{t['name']} · #{t['id']}": t["id"] for t in tenant_list}
        default_label = next(iter(options))
        label = st.selectbox("当前租户", list(options), index=0, help="所有页面都按租户隔离")
        st.session_state["tenant_id"] = options[label]
        st.session_state["tenant_name"] = tenant_list[
            list(options.values()).index(options[label])
        ]["name"]
    else:
        st.session_state["tenant_id"] = 1
        st.session_state["tenant_name"] = "未初始化"
        st.warning("还没有租户数据,请先运行: python scripts/import_knowledge_seed.py")

    st.divider()
    st.caption("数据每 30 秒自动刷新一次;也可点击页面右上角手动刷新。")

pages = [
    st.Page("pages/overview.py", title="总览", icon="📈", default=True),
    st.Page("pages/data_browser.py", title="数据浏览", icon="🗂️"),
    st.Page("pages/ingestion_manager.py", title="摄取管理", icon="📥"),
    st.Page("pages/ingestion_traces.py", title="摄取追踪", icon="🧭"),
    st.Page("pages/query_traces.py", title="查询追踪", icon="🔍"),
    st.Page("pages/evaluation_panel.py", title="评估面板", icon="🎯"),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()
