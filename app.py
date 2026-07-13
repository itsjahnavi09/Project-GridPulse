import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = BASE_DIR / "individual_household_power_consumption.zip"
TXT_PATH = DATA_DIR / "household_power_consumption.txt"


def ensure_dataset() -> None:
    """Download and extract the dataset when it is not already available."""
    if TXT_PATH.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        urlretrieve(DATA_URL, ZIP_PATH)

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)


def load_and_prepare_data():
    ensure_dataset()

    df = pd.read_csv(
        TXT_PATH,
        sep=";",
        na_values="?",
        low_memory=False,
    )

    key_cols = ["Date", "Time", "Global_active_power"]
    df = df.dropna(subset=key_cols).copy()

    num_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["Date", "Time"], keep="first")

    df = df[
        (df["Global_active_power"] >= 0)
        & (df["Global_intensity"] >= 0)
        & (df["Voltage"] > 0)
    ].copy()

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["date_only"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["day_name"] = df["datetime"].dt.day_name()
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6])

    def power_category(value):
        if value < 1:
            return "Low"
        if value < 3:
            return "Medium"
        return "High"

    df["Power_Category"] = df["Global_active_power"].apply(power_category)

    daily_total = (
        df.groupby("date_only")["Global_active_power"].sum() / 60
    )
    hourly_avg = df.groupby("hour")["Global_active_power"].mean().sort_index()

    peak_hour = int(hourly_avg.idxmax())
    offpeak_hour = int(hourly_avg.idxmin())

    weekday_avg = df.loc[~df["is_weekend"], "Global_active_power"].mean()
    weekend_avg = df.loc[df["is_weekend"], "Global_active_power"].mean()

    peak_window = [(peak_hour - 1) % 24, peak_hour, (peak_hour + 1) % 24]
    peak_usage_kwh = (
        df.loc[df["hour"].isin(peak_window), "Global_active_power"].sum() / 60
    )

    peak_rate = 8
    offpeak_rate = 5
    peak_cost = peak_usage_kwh * peak_rate
    offpeak_cost = peak_usage_kwh * offpeak_rate
    money_saved = peak_cost - offpeak_cost

    return {
        "df": df,
        "daily_total": daily_total,
        "hourly_avg": hourly_avg,
        "peak_hour": peak_hour,
        "offpeak_hour": offpeak_hour,
        "weekday_avg": weekday_avg,
        "weekend_avg": weekend_avg,
        "peak_usage_kwh": peak_usage_kwh,
        "peak_cost": peak_cost,
        "offpeak_cost": offpeak_cost,
        "money_saved": money_saved,
    }


DATA = load_and_prepare_data()


def show_dashboard():
    df = DATA["df"]
    daily_total = DATA["daily_total"]
    hourly_avg = DATA["hourly_avg"]

    insights = f"""
Cleaned Dataset Shape: {df.shape}

Peak Hour: {DATA['peak_hour']}:00
Off-Peak Hour: {DATA['offpeak_hour']}:00

Average Weekday Consumption: {DATA['weekday_avg']:.2f} kW
Average Weekend Consumption: {DATA['weekend_avg']:.2f} kW

Most Common Power Category: {df['Power_Category'].value_counts().idxmax()}

Estimated Peak-Window Usage: {DATA['peak_usage_kwh']:.2f} kWh
Estimated Cost at Peak Rate: ₹{DATA['peak_cost']:.2f}
Estimated Cost at Off-Peak Rate: ₹{DATA['offpeak_cost']:.2f}
Estimated Money Saved: ₹{DATA['money_saved']:.2f}

Conclusion:
Shifting heavy appliance usage from peak to off-peak hours can reduce
electricity cost and improve energy efficiency.
""".strip()

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(daily_total.index, daily_total.values)
    ax1.set_title("Daily Electricity Consumption Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Energy Consumption (kWh)")
    fig1.autofmt_xdate()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(hourly_avg.index, hourly_avg.values, marker="o")
    ax2.set_title("Average Electricity Consumption by Hour")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Average Power (kW)")
    ax2.grid(True)
    fig2.tight_layout()

    category_counts = df["Power_Category"].value_counts()
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct="%1.1f%%",
    )
    ax3.set_title("Power Category Distribution")
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.bar(
        ["Weekday", "Weekend"],
        [DATA["weekday_avg"], DATA["weekend_avg"]],
    )
    ax4.set_title("Weekday vs Weekend Consumption")
    ax4.set_ylabel("Average Power (kW)")
    fig4.tight_layout()

    preview_columns = [
        "datetime",
        "Global_active_power",
        "Voltage",
        "Global_intensity",
        "Power_Category",
    ]

    return insights, fig1, fig2, fig3, fig4, df[preview_columns].head(10)


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Electricity Consumption Pattern Analysis",
) as demo:
    gr.Markdown(
        """
        # ⚡ Electricity Consumption Pattern Analysis
        ### Fundamentals of Data Analytics Project Dashboard

        This dashboard shows the final results of data cleaning,
        preprocessing, analysis and cost-saving estimation.
        """
    )

    run_button = gr.Button("📊 Show Final Dashboard", variant="primary")
    insights_output = gr.Textbox(
        label="📌 Insights and Summary",
        lines=18,
    )

    gr.Markdown("## 📈 Visual Analysis")

    with gr.Row():
        chart1_output = gr.Plot(label="Daily Trend")
        chart2_output = gr.Plot(label="Hourly Consumption")

    with gr.Row():
        chart3_output = gr.Plot(label="Power Category Distribution")
        chart4_output = gr.Plot(label="Weekday vs Weekend")

    gr.Markdown("## 🧾 Processed Data Preview")
    table_output = gr.Dataframe(label="Top Processed Records")

    run_button.click(
        fn=show_dashboard,
        inputs=[],
        outputs=[
            insights_output,
            chart1_output,
            chart2_output,
            chart3_output,
            chart4_output,
            table_output,
        ],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
