import os
from dotenv import load_dotenv

load_dotenv()

REPORT_MODE = os.getenv("REPORT_MODE", "static")

if REPORT_MODE == "agent":
    from app.services.report.agent_report import generate_report
else:
    from app.services.report.static_report import generate_report

__all__ = ["generate_report"]