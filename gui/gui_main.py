import sys
import os
import numpy as np
from datetime import datetime
from collections import deque
from joblib import load
import matplotlib.pyplot as plt
import csv
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
    QProgressBar, QTextEdit, QTabWidget, QComboBox, QSlider, QHBoxLayout, QSpinBox, QGroupBox,
    QDateEdit, QCheckBox
)
from PyQt5.QtCore import Qt, QSettings, QTimer, QDate

from core.inference import predict_signal_from_mt5
from core.mt5_utils import initialize_mt5, fetch_historical_data
from core.charting import plot_chart_with_levels
from core.strategies import ma_crossover_signal, macd_signal_strategy, pattern_trigger_strategy



TIMEFRAME = "M15"
# CAPITAL removed as constant; it will now be user-defined from config or UI

class ForexDashboard(QWidget):
    def __init__(self):
        self.hud_minimized = False

        super().__init__()
        self.setWindowTitle("📊 Forex Signal Dashboard")
        self.setGeometry(200, 200, 1000, 800)

        initialize_mt5()
        self.settings = QSettings("TradeSense", "ForexGUI")
        self.dark_toggle = QCheckBox("🌙 Dark Mode")
        self.dark_toggle.setChecked(self.settings.value("dark_mode", True, type=bool))

        # Rolling confidence tracking
        self.conf_history = deque(maxlen=20)
        self.conf_timestamps = deque(maxlen=20)

        self.auto_run_enabled = self.settings.value("auto_run_enabled", True, type=bool)
        self.initUI()

        # Scheduled auto-analysis toggle
        self.auto_run_enabled = self.settings.value("auto_run_enabled", True, type=bool)
        self.interval_minutes = self.settings.value("auto_interval", 15, type=int)
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.run_analysis)
        if self.auto_run_enabled:
            self.interval_minutes = self.settings.value("auto_interval", 15, type=int)
            self.analysis_timer.start(self.interval_minutes * 60 * 1000)

        # Countdown timer for display
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)  # every second
        self.seconds_left = self.interval_minutes * 60

    def init_strategy_builder_tab(self):
        layout = QVBoxLayout()

        self.logic_text = QTextEdit()
        self.logic_text.setPlaceholderText("e.g., if rsi < 30 and macd > 0: signal = 1")
        layout.addWidget(QLabel("📜 Define Strategy Logic:"))
        layout.addWidget(self.logic_text)

        self.save_strategy_button = QPushButton("💾 Save Strategy Stub")
        self.save_strategy_button.clicked.connect(self.save_strategy_stub)
        layout.addWidget(self.save_strategy_button)

        self.strategy_builder_tab.setLayout(layout)

    def initUI(self):
        layout = QVBoxLayout()
        self.countdown_label = QLabel("Next auto-analysis in: --:--")

        # --- Strategy Settings Panel ---
        self.settings_box = QGroupBox("⚙️ Strategy Settings")
        self.settings_box.setCheckable(True)
        self.settings_box.setChecked(True)
        settings_layout = QHBoxLayout()

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(30)
        self.threshold_slider.setMaximum(90)
        self.threshold_slider.setValue(self.settings.value("threshold", 60, type=int))
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_label = QLabel(f"Confidence Threshold: {self.threshold_slider.value()}")
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_label.setText(f"Confidence Threshold: {v}"))

        self.sl_tp_ratio = QComboBox()
        self.sl_tp_ratio.addItems(["1:1", "1:1.5", "1:2", "1:3"])
        self.sl_tp_ratio.setCurrentText(self.settings.value("sl_tp_ratio", "1:2"))
        self.sl_tp_label = QLabel("SL/TP Ratio")

        self.candle_spin = QSpinBox()
        self.candle_spin.setRange(50, 500)
        self.candle_spin.setValue(self.settings.value("candles", 100, type=int))
        self.candle_label = QLabel("Candles")

        self.symbol_selector = QComboBox()
        self.symbol_selector.addItems(["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "XAUUSD", "BTCUSD"])
        layout.addWidget(QLabel("Select Symbol:"))
        layout.addWidget(self.symbol_selector)

        
        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        self.timeframe_box.setCurrentText(self.settings.value("timeframe", TIMEFRAME))
        self.timeframe_label = QLabel("Timeframe")

        self.capital_spin = QSpinBox()
        self.capital_spin.setRange(100, 100000)
        self.capital_spin.setValue(self.settings.value("capital", 1000, type=int))
        self.capital_label = QLabel("Capital ($)")

        self.start_date_label = QLabel("Backtest Start Date")
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDate(QDate.currentDate().addDays(-30))
        self.start_date_edit.setCalendarPopup(True)

        self.end_date_label = QLabel("Backtest End Date")
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setCalendarPopup(True)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 180)
        self.interval_spin.setValue(self.settings.value("auto_interval", 15, type=int))
        self.interval_spin.valueChanged.connect(lambda v: self.settings.setValue("auto_interval", v))
        self.strategy_label = QLabel("Strategy")
        self.strategy_box = QComboBox()
        self.strategy_box.addItems(["Default Model", "MA Crossover", "MACD Signal", "Pattern Trigger"])
        self.strategy_box.setCurrentText(self.settings.value("strategy", "Default Model"))
        self.strategy_box.currentTextChanged.connect(lambda s: self.settings.setValue("strategy", s))

        self.interval_label = QLabel("Auto Run Interval (min)")
        settings_layout.addWidget(self.start_date_label)
        settings_layout.addWidget(self.start_date_edit)
        settings_layout.addWidget(self.end_date_label)
        settings_layout.addWidget(self.end_date_edit)

        self.auto_run_toggle = QCheckBox("🕒 Enable Auto Analysis")
        self.auto_run_toggle.setChecked(self.auto_run_enabled)
        self.auto_run_toggle.stateChanged.connect(self.toggle_auto_analysis)

        settings_layout.addWidget(self.threshold_label)
        settings_layout.addWidget(self.threshold_slider)
        settings_layout.addWidget(self.sl_tp_label)
        settings_layout.addWidget(self.sl_tp_ratio)
        settings_layout.addWidget(self.candle_label)
        settings_layout.addWidget(self.candle_spin)
        settings_layout.addWidget(self.timeframe_label)
        settings_layout.addWidget(self.timeframe_box)
        settings_layout.addWidget(self.capital_label)
        settings_layout.addWidget(self.capital_spin)
        settings_layout.addWidget(self.strategy_label)
        settings_layout.addWidget(self.strategy_box)
        settings_layout.addWidget(self.interval_label)
        settings_layout.addWidget(self.interval_spin)
        settings_layout.addWidget(self.auto_run_toggle)

                # 🌙 Theme Toggle
        self.dark_toggle = QCheckBox("🌙 Dark Mode")
        self.dark_toggle.setChecked(self.settings.value("dark_mode", True, type=bool))
        self.dark_toggle.stateChanged.connect(self.apply_theme)
        settings_layout.addWidget(self.dark_toggle)

        self.settings_box.setLayout(settings_layout)
        layout.addWidget(self.countdown_label)
        layout.addWidget(self.settings_box)

        # --- Signal Output Area ---
        self.signal_label = QLabel("Signal: ❓", self)
        self.signal_label.setAlignment(Qt.AlignCenter)
        self.signal_label.setStyleSheet("font-size: 28px;")

        self.confidence_bar = QProgressBar(self)
        self.confidence_bar.setMaximum(100)

        self.sltp_box = QTextEdit(self)
        self.sltp_box.setReadOnly(True)
        self.sltp_box.setFixedHeight(60)

        self.refresh_button = QPushButton("🔁 Run Analysis", self)
        self.refresh_button.clicked.connect(self.run_analysis)

        layout.addWidget(self.signal_label)
        layout.addWidget(self.confidence_bar)
        layout.addWidget(QLabel("Suggested SL/TP:"))
        layout.addWidget(self.sltp_box)
        layout.addWidget(self.refresh_button)


        self.hud_toggle_button = QPushButton("🗕 Minimize HUD")
        self.hud_toggle_button.clicked.connect(self.toggle_hud_mode)
        layout.addWidget(self.hud_toggle_button)

        # --- Tabs ---
        self.tabs = QTabWidget()
        self.journal_text = QTextEdit()
        self.telegram_token_input = QTextEdit()
        self.telegram_token_input.setPlaceholderText("Enter your TELEGRAM_TOKEN here")
        self.telegram_token_input.setFixedHeight(30)
        self.telegram_token_input.setText(self.settings.value("telegram_token", ""))
        if self.telegram_token_input.toPlainText().strip():
            self.telegram_token_input.hide()
        

        self.telegram_chat_input = QTextEdit()
        self.telegram_chat_input.setPlaceholderText("Enter your TELEGRAM_CHAT_ID here")
        self.telegram_chat_input.setFixedHeight(30)
        self.telegram_chat_input.setText(self.settings.value("telegram_chat_id", ""))
        if self.telegram_chat_input.toPlainText().strip():
            self.telegram_chat_input.hide()
        

        self.save_alert_btn = QPushButton("💾 Save Telegram Config")
        self.save_alert_btn.clicked.connect(self.save_telegram_config)

        self.test_alert_btn = QPushButton("📨 Send Test Alert")
        self.test_alert_btn.clicked.connect(self.send_test_alert)

        self.alert_status_label = QLabel("")
        self.journal_text.setReadOnly(True)
        self.tabs.currentChanged.connect(self.refresh_journal_tab)
        layout.addWidget(self.tabs)

        # Chart Tab
        
        self.chart_fig, self.ax = plt.subplots()
        self.chart_fig.tight_layout()
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.canvas = FigureCanvas(self.chart_fig)
        self.chart_toolbar = NavigationToolbar2QT(self.canvas, self)
        chart_tab = QWidget()
        chart_layout = QVBoxLayout()
        chart_layout.addWidget(self.chart_toolbar)
        chart_layout.addWidget(self.canvas)
        chart_tab.setLayout(chart_layout)
        self.tabs.addTab(chart_tab, "📈 Chart")

        # Feature Importance Tab
        self.importance_fig, self.importance_ax = plt.subplots(figsize=(5, 3))
        self.importance_fig.tight_layout()
        self.importance_canvas = FigureCanvas(self.importance_fig)
        importance_tab = QWidget()
        importance_layout = QVBoxLayout()
        importance_layout.addWidget(self.importance_canvas)
        importance_tab.setLayout(importance_layout)
        self.tabs.addTab(importance_tab, "📊 Feature Importance")

        # Confidence History Tab
        self.history_fig, self.history_ax = plt.subplots(figsize=(6, 3))
        self.history_fig.tight_layout()
        self.history_canvas = FigureCanvas(self.history_fig)
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        history_layout.addWidget(self.history_canvas)
        history_tab.setLayout(history_layout)
        self.tabs.addTab(history_tab, "📉 Confidence History")

        # Backtest Tab
        self.backtest_fig, self.backtest_ax = plt.subplots()
        self.backtest_fig.tight_layout()
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.backtest_canvas = FigureCanvas(self.backtest_fig)
        self.backtest_toolbar = NavigationToolbar2QT(self.backtest_canvas, self)
        self.backtest_btn = QPushButton("▶️ Run Backtest")
        self.backtest_btn.clicked.connect(self.run_backtest_tab)
        backtest_tab = QWidget()
        backtest_layout = QVBoxLayout()
        backtest_layout.addWidget(self.backtest_btn)
        backtest_layout.addWidget(self.backtest_toolbar)
        backtest_layout.addWidget(self.backtest_canvas)
        backtest_tab.setLayout(backtest_layout)
        self.tabs.addTab(backtest_tab, "🧪 Backtest")

        # Trade Journal Tab
        journal_tab = QWidget()
        journal_layout = QVBoxLayout()
        journal_layout.addWidget(self.journal_text)
        journal_tab.setLayout(journal_layout)
        self.tabs.addTab(journal_tab, "📄 Trade Journal")

                # Alerts Tab
        alerts_tab = QWidget()
        alerts_layout = QVBoxLayout()
        alerts_layout.addWidget(QLabel("Telegram Bot Token:"))
        alerts_layout.addWidget(self.telegram_token_input)
        if self.telegram_token_input.toPlainText().strip():
            self.telegram_token_input.hide()
            self.telegram_token_toggle = QPushButton("✏️ Edit Token")
            self.telegram_token_toggle.clicked.connect(lambda: self.telegram_token_input.show())
            alerts_layout.addWidget(self.telegram_token_toggle)
        alerts_layout.addWidget(QLabel("Telegram Chat ID:"))
        alerts_layout.addWidget(self.telegram_chat_input)
        if self.telegram_chat_input.toPlainText().strip():
            self.telegram_chat_input.hide()
            self.telegram_chat_toggle = QPushButton("✏️ Edit Chat ID")
            self.telegram_chat_toggle.clicked.connect(lambda: self.telegram_chat_input.show())
            alerts_layout.addWidget(self.telegram_chat_toggle)
        alerts_layout.addWidget(self.save_alert_btn)
        alerts_layout.addWidget(self.test_alert_btn)
        alerts_layout.addWidget(self.alert_status_label)

        # Email alert inputs
        alerts_layout.addWidget(QLabel("SMTP Email Address:"))
        self.smtp_email_input = QTextEdit()
        self.smtp_email_input.setPlaceholderText("example@gmail.com")
        self.smtp_email_input.setFixedHeight(30)
        self.smtp_email_input.setText(self.settings.value("smtp_email", ""))
        if self.smtp_email_input.toPlainText().strip():
            self.smtp_email_input.hide()
            self.smtp_email_toggle = QPushButton("✏️ Edit Email")
            self.smtp_email_toggle.clicked.connect(lambda: self.smtp_email_input.show())
            alerts_layout.addWidget(self.smtp_email_toggle)
        alerts_layout.addWidget(self.smtp_email_input)

        alerts_layout.addWidget(QLabel("SMTP Password (App Password):"))
        self.smtp_pass_input = QTextEdit()
        self.smtp_pass_input.setPlaceholderText("Your app-specific password")
        self.smtp_pass_input.setFixedHeight(30)
        self.smtp_pass_input.setText(self.settings.value("smtp_pass", ""))
        if self.smtp_pass_input.toPlainText().strip():
            self.smtp_pass_input.hide()
            self.smtp_pass_toggle = QPushButton("✏️ Edit App Password")
            self.smtp_pass_toggle.clicked.connect(lambda: self.smtp_pass_input.show())
            alerts_layout.addWidget(self.smtp_pass_toggle)
        alerts_layout.addWidget(self.smtp_pass_input)

        alerts_layout.addWidget(QLabel("Recipient Email:"))
        self.smtp_to_input = QTextEdit()
        self.smtp_to_input.setPlaceholderText("To address")
        self.smtp_to_input.setFixedHeight(30)
        self.smtp_to_input.setText(self.settings.value("smtp_to", ""))
        if self.smtp_to_input.toPlainText().strip():
            self.smtp_to_input.hide()
            self.smtp_to_toggle = QPushButton("✏️ Edit Recipient")
            self.smtp_to_toggle.clicked.connect(lambda: self.smtp_to_input.show())
            alerts_layout.addWidget(self.smtp_to_toggle)
        alerts_layout.addWidget(self.smtp_to_input)

        self.save_email_btn = QPushButton("💾 Save Email Config")
        self.save_email_btn.clicked.connect(self.save_email_config)
        alerts_layout.addWidget(self.save_email_btn)

        self.test_email_btn = QPushButton("📧 Send Test Email")
        self.test_email_btn.clicked.connect(self.send_test_email)
        alerts_layout.addWidget(self.test_email_btn)
        alerts_tab.setLayout(alerts_layout)
        self.tabs.addTab(alerts_tab, "📬 Alerts")

        # --- Live Feed Tab ---
        self.live_feed_box = QTextEdit()
        self.live_feed_box.setReadOnly(True)
        self.live_feed_box.setPlaceholderText("Live signal feed will appear here with timestamps...")
        live_tab = QWidget()
        live_layout = QVBoxLayout()
        live_layout.addWidget(self.live_feed_box)
        live_tab.setLayout(live_layout)
        self.tabs.addTab(live_tab, "📡 Live Feed")

        

        self.strategy_builder_tab = QWidget()
        self.init_strategy_builder_tab()
        self.tabs.addTab(self.strategy_builder_tab, "🧠 Strategy Builder")


        self.levels_box = QTextEdit(self)
        self.levels_box.setReadOnly(True)
        self.levels_box.setFixedHeight(80)
        layout.addWidget(QLabel("📐 Entry / Resistance / Support"))
        layout.addWidget(self.levels_box)

        self.setLayout(layout)

    


    def toggle_hud_mode(self):
      self.hud_minimized = not self.hud_minimized

      show = not self.hud_minimized
      self.canvas.setVisible(show)
      self.sltp_box.setVisible(show)
      self.refresh_button.setVisible(show)
      self.importance_canvas.setVisible(show)
      self.live_feed_box.setVisible(show)
      self.tabs.setVisible(show) if hasattr(self, 'tabs') else None

      self.hud_toggle_button.setText("🗖 Expand HUD" if self.hud_minimized else "🗕 Minimize HUD")

    def run_backtest_tab(self):
        tf = self.timeframe_box.currentText()
        threshold = self.threshold_slider.value() / 100
        candle_count = self.candle_spin.value()
        sl_mult, tp_mult = map(float, self.sl_tp_ratio.currentText().split(':'))

        from core.feature_engineering import generate_features
        from core.model_utils import load_model

        from PyQt5.QtCore import QDate
        start_dt = self.start_date_edit.date().toPyDate()
        end_dt = self.end_date_edit.date().toPyDate()
        SYMBOL = self.symbol_selector.currentText()
        df = fetch_historical_data(SYMBOL, tf, candle_count + 50)
        df = df[(df['time'].dt.date >= start_dt) & (df['time'].dt.date <= end_dt)]
        df = generate_features(df)

        if len(df) < 24:
            self.backtest_ax.clear()
            self.backtest_ax.set_title("❌ Not enough historical data for this timeframe.")
            self.backtest_canvas.draw()
            return

        selected_strategy = self.strategy_box.currentText()
        model = load_model() if selected_strategy == "Default Model" else None
        capital = self.settings.value("capital", 1000, type=int)
        equity = [capital]
        equity_array = np.array(equity)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = running_max - equity_array
        max_drawdown = drawdown.max()
        wins = 0
        trades = 0

        for i in range(20, len(df) - 3):
            row = df.iloc[i]
            features = row[['rsi', 'macd', 'ema_fast', 'ema_slow', 'atr']].astype(float).to_frame().T
            if selected_strategy == "Default Model":
                prob = model.predict_proba(features)[0, 1]
            elif selected_strategy == "MA Crossover":
                prob = 1 if row['ema_fast'] > row['ema_slow'] else 0
            elif selected_strategy == "MACD Signal":
                prob = 1 if row['macd'] > 0 else 0
            elif selected_strategy == "Pattern Trigger":
                prob = 1 if row['close'] > row['open'] else 0
            else:
                prob = 0
            if prob > threshold:
                price = row['close']
                atr = row['atr']
                sl = price - atr * sl_mult
                tp = price + atr * tp_mult

                future_high = df.iloc[i+1:i+4]['high'].max()
                future_low = df.iloc[i+1:i+4]['low'].min()

                trades += 1
                if future_high >= tp:
                    wins += 1
                    equity.append(equity[-1] + 10)
                elif future_low <= sl:
                    equity.append(equity[-1] - 10)
                else:
                    equity.append(equity[-1])

        winrate = (wins / trades * 100) if trades > 0 else 0
        pnl = equity[-1] - capital

        self.backtest_ax.clear()
        self.backtest_ax.plot(equity, color='blue', label="Equity")
        self.backtest_ax.plot(drawdown, color='red', label="Drawdown")
        self.backtest_ax.set_xlabel("Trade #")
        self.backtest_ax.set_ylabel("Equity / Drawdown")
        self.backtest_ax.set_title(f"Equity Curve\nWinrate: {winrate:.2f}% | PnL: {pnl:.2f} | Max Drawdown: {max_drawdown:.2f}")

        self.backtest_ax.legend()
        self.backtest_ax.grid(True)
        self.backtest_canvas.draw()

    
    from core.strategies import ma_crossover_signal, macd_signal_strategy, pattern_trigger_strategy

    def run_analysis(self):
        tf = self.timeframe_box.currentText()
        threshold = self.threshold_slider.value() / 100
        candle_count = self.candle_spin.value()

        self.settings.setValue("threshold", self.threshold_slider.value())
        self.settings.setValue("sl_tp_ratio", self.sl_tp_ratio.currentText())
        self.settings.setValue("candles", candle_count)
        self.settings.setValue("timeframe", tf)
        self.settings.setValue("capital", self.capital_spin.value())

        capital = self.capital_spin.value()
        selected_strategy = self.strategy_box.currentText()
        if selected_strategy == "MA Crossover":
            SYMBOL = self.symbol_selector.currentText()
            result = ma_crossover_signal(SYMBOL, tf, capital)
        elif selected_strategy == "MACD Signal":
            SYMBOL = self.symbol_selector.currentText()
            result = macd_signal_strategy(SYMBOL, tf, capital)
        elif selected_strategy == "Pattern Trigger":
            SYMBOL = self.symbol_selector.currentText()
            result = pattern_trigger_strategy(SYMBOL, tf, capital)
        else:
            SYMBOL = self.symbol_selector.currentText()
            result = predict_signal_from_mt5(symbol=SYMBOL, timeframe=tf, capital=capital)

        signal = 1 if result["confidence"] > threshold else 0
        result["signal"] = signal

        # --- Apply SL/TP logic based on ratio ---
        sl_mult, tp_mult = map(float, self.sl_tp_ratio.currentText().split(':'))
        atr = result.get("atr", 0.0015)
        price = result.get("price", 1.0000)
        result["stop_loss"] = round(price - atr * sl_mult, 5)
        result["take_profit"] = round(price + atr * tp_mult, 5)

        if signal:
            self.signal_label.setText("✅ GOOD TRADE SIGNAL")
            self.signal_label.setStyleSheet("color: green; font-size: 28px;")
        else:
            self.signal_label.setText("❌ NO TRADE SIGNAL")
            self.signal_label.setStyleSheet("color: red; font-size: 28px;")

        self.confidence_bar.setValue(int(result["confidence"] * 100))
        self.sltp_box.setText(f"Stop Loss: {result['stop_loss']}\nTake Profit: {result['take_profit']}")

        # --- Chart ---
        self.ax.clear()
        chart_path = result.get("chart_path", None)
        if chart_path and os.path.exists(chart_path):
            img = plt.imread(chart_path)
            self.ax.imshow(img)
            self.ax.set_title("Latest Chart with Overlays")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
        self.canvas.draw()

        # --- Feature Importance ---
        model = load("models/forex_model.pkl")
        features = ['rsi', 'macd', 'ema_fast', 'ema_slow', 'atr']
        importances = model.feature_importances_
        total = sum(importances)
        importances = [(imp / total) * 100 for imp in importances] if total > 0 else [0] * len(features)
        sorted_data = sorted(zip(features, importances), key=lambda x: x[1])
        labels, values = zip(*sorted_data)

        self.importance_ax.clear()
        bars = self.importance_ax.barh(labels, values, color="purple")
        self.importance_ax.set_xlabel("Importance (%)")
        self.importance_ax.set_title("Model Feature Importances")
        self.importance_ax.set_xlabel("Importance (%)")
        self.importance_ax.legend(["Higher = More impact on prediction"], loc="lower right", fontsize=8, frameon=False)
        for bar, val in zip(bars, values):
            self.importance_ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va='center')
        self.importance_canvas.draw()

        # --- Trade Journal Logging ---
        log_row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), SYMBOL, signal, round(result["confidence"], 4), price, result["stop_loss"], result["take_profit"]]
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", "trade_journal.csv")
        write_header = not os.path.exists(log_path)
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Time", "Pair", "Signal", "Confidence", "Price", "Stop Loss", "Take Profit"])
            writer.writerow(log_row)

        # --- Telegram Alert ---
        try:
            with open(chart_path, "rb") as img_file:
                files = {"photo": img_file}
                telegram_token = self.settings.value("telegram_token", type=str)
                telegram_chat_id = self.settings.value("telegram_chat_id", type=str)
                if telegram_token and telegram_chat_id:
                    msg = (
    f"{'✅ BUY' if signal else '❌ NO TRADE'}\n"
    f"Confidence: {int(result['confidence'] * 100)}%\n"
    f"Price: {price}\n"
    f"SL: {result['stop_loss']} | TP: {result['take_profit']}"
)


                    
                    import requests
                    requests.post(f"https://api.telegram.org/bot{telegram_token}/sendPhoto", data={"chat_id": telegram_chat_id, "caption": msg}, files=files)
        except Exception as e:
            print(f"[Telegram Error] {e}")

        # --- Email Alert ---
        try:
            from email.mime.text import MIMEText
            import smtplib
            email = self.settings.value("smtp_email", type=str)
            password = self.settings.value("smtp_pass", type=str)
            to = self.settings.value("smtp_to", type=str)
            if email and password and to:
                msg = (
    f"{'✅ BUY' if signal else '❌ NO TRADE'}\n"
    f"Confidence: {int(result['confidence'] * 100)}%\n"
    f"Price: {price}\n"
    f"SL: {result['stop_loss']} | TP: {result['take_profit']}"
)

                msg['Subject'] = '📈 Forex Signal Alert'
                msg['From'] = email
                msg['To'] = to
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                server.login(email, password)
                server.send_message(msg)
                server.quit()
        except Exception as e:
            print(f"[Email Error] {e}")

        # --- Confidence History ---
        conf_pct = result["confidence"] * 100
        self.live_feed_box.append(f"{datetime.now().strftime('%H:%M:%S')} | {'✅ BUY' if signal else '❌ NO TRADE'} | Confidence: {int(conf_pct)}%")
        conf_pct = result["confidence"] * 100
        self.conf_history.append(conf_pct)
        self.conf_timestamps.append(datetime.now().strftime("%H:%M:%S"))

        self.history_ax.clear()
        x = list(range(len(self.conf_history)))
        y = list(self.conf_history)

        if len(x) > 1:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = ["red" if (y[i] + y[i+1]) / 2 < 50 else "orange" if (y[i] + y[i+1]) / 2 < 70 else "green" for i in range(len(y)-1)]
            lc = LineCollection(segments, colors=colors, linewidths=2)
            self.history_ax.add_collection(lc)

        for i, conf in enumerate(y):
            signal = 1 if conf >= threshold * 100 else 0
            dot_color = 'green' if signal else 'red'
            self.history_ax.plot(x[i], y[i], 'o', color=dot_color, markersize=5)

        if y:
            self.history_ax.axhline(y=max(y), color='gray', linestyle='--', linewidth=1, label=f"Max: {max(y):.1f}%")
            self.history_ax.axhline(y=min(y), color='lightgray', linestyle='--', linewidth=1, label=f"Min: {min(y):.1f}%")

        self.history_ax.set_ylim(0, 100)
        self.history_ax.set_ylabel("Confidence %")
        self.history_ax.set_title("Rolling Signal Confidence")
        self.history_ax.set_xticks(x)
        self.history_ax.set_xticklabels([self.conf_timestamps[i] for i in x], rotation=45, ha="right", fontsize=8)
        self.history_ax.legend(fontsize=7)
        self.history_canvas.draw()

        entry = result.get("entry_price", "N/A")
        res = result.get("resistance", "N/A")
        sup = result.get("support", "N/A")
        self.levels_box.setText(
            f"📌 Entry: {entry}\n📈 Resistance: {res}\n📉 Support: {sup}"
        )


    def refresh_journal_tab(self, index):
        if self.tabs.tabText(index) == "📄 Trade Journal":
            path = os.path.join("logs", "trade_journal.csv")
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.journal_text.setPlainText(f.read())
            else:
                self.journal_text.setPlainText("No trades logged yet.")

    def save_telegram_config(self):
        token = self.telegram_token_input.toPlainText().strip()
        chat_id = self.telegram_chat_input.toPlainText().strip()
        self.settings.setValue("telegram_token", token)
        self.telegram_token_input.hide()
        self.telegram_chat_input.hide()
        self.settings.setValue("telegram_chat_id", chat_id)

    def send_test_alert(self):
        import requests
        token = self.telegram_token_input.toPlainText().strip()
        chat_id = self.telegram_chat_input.toPlainText().strip()
        if not token or not chat_id:
            self.alert_status_label.setText("❌ Missing token or chat_id")
            return
        try:
            msg = "📈 Test Alert from Forex Dashboard\nIf you received this, your Telegram is working!"

            resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": msg})
            if resp.status_code == 200:
                self.alert_status_label.setText("✅ Test alert sent successfully!")
            else:
                self.alert_status_label.setText(f"⚠️ Error: {resp.status_code}")
        except Exception as e:
            self.alert_status_label.setText(f"❌ Failed to send: {e}")

    def save_email_config(self):
        self.settings.setValue("smtp_email", self.smtp_email_input.toPlainText().strip())
        self.settings.setValue("smtp_pass", self.smtp_pass_input.toPlainText().strip())
        self.settings.setValue("smtp_to", self.smtp_to_input.toPlainText().strip())
        self.smtp_email_input.hide()
        self.smtp_pass_input.hide()
        self.smtp_to_input.hide()

    def send_test_email(self):
        import smtplib
        from email.mime.text import MIMEText
        email = self.smtp_email_input.toPlainText().strip()
        password = self.smtp_pass_input.toPlainText().strip()
        to = self.smtp_to_input.toPlainText().strip()

        if not all([email, password, to]):
            self.alert_status_label.setText("❌ Missing email config")
            return
        try:
            msg = MIMEText("📈 Test Email Alert from Forex Dashboard\nIf you received this, SMTP config works!")

            msg['Subject'] = 'Forex Test Alert'
            msg['From'] = email
            msg['To'] = to

            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(email, password)
            server.send_message(msg)
            server.quit()
            self.alert_status_label.setText("✅ Test email sent successfully!")
        except Exception as e:
            self.alert_status_label.setText(f"❌ Email failed: {e}")

    def toggle_auto_analysis(self, state):
        enabled = bool(state)
        self.settings.setValue("auto_run_enabled", enabled)
        if enabled:
            interval = self.interval_spin.value()
            self.seconds_left = interval * 60
            self.analysis_timer.start(interval * 60 * 1000)
        else:
            self.analysis_timer.stop()
            self.countdown_label.setText("⏸ Auto-analysis paused")

    def update_countdown(self):
        if self.auto_run_toggle.isChecked() and self.analysis_timer.isActive():
            self.seconds_left -= 1
            if self.seconds_left <= 0:
                interval = self.interval_spin.value()
                self.seconds_left = interval * 60
            mins, secs = divmod(self.seconds_left, 60)
            self.countdown_label.setText(f"Next auto-analysis in: {mins:02d}:{secs:02d}")


    def save_strategy_stub(self):
        logic = self.logic_text.toPlainText().strip()
        if logic:
            os.makedirs("strategies", exist_ok=True)
            with open("strategies/custom_logic.py", "w") as f:
                f.write(logic)
            self.live_feed_box.append("🧠 Custom strategy logic saved to strategies/custom_logic.py.")
        else:
            self.live_feed_box.append("⚠️ No logic to save.")



    def apply_theme(self):
        is_dark = self.dark_toggle.isChecked()
        self.settings.setValue("dark_mode", is_dark)

        if is_dark:
            self.setStyleSheet("""
                QWidget { background-color: #121212; color: #FFFFFF; }
                QGroupBox { border: 1px solid #333; margin-top: 10px; }
                QGroupBox:title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; color: #CCCCCC; }
                QPushButton { background-color: #2D2D2D; border: 1px solid #444; padding: 5px; }
                QPushButton:hover { background-color: #3C3C3C; }
                QProgressBar { background-color: #2A2A2A; border: 1px solid #444; text-align: center; color: white; }
                QTabWidget::pane { border: 1px solid #444; }
                QTabBar::tab { background: #2A2A2A; border: 1px solid #444; padding: 5px; }
                QTabBar::tab:selected { background: #444; }
                QComboBox, QSpinBox, QSlider, QTextEdit { background-color: #1E1E1E; color: white; border: 1px solid #444; }
                QLabel { color: #CCCCCC; }
            """)
        else:
            self.setStyleSheet("")




if __name__ == "__main__":
    app = QApplication(sys.argv)

    from PyQt5.QtWidgets import QCheckBox
    is_dark = QSettings("TradeSense", "ForexGUI").value("dark_mode", True, type=bool)

    if is_dark:
        app.setStyleSheet("""
            QWidget { background-color: #121212; color: #FFFFFF; }
            QGroupBox { border: 1px solid #333; margin-top: 10px; }
            QGroupBox:title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; color: #CCCCCC; }
            QPushButton { background-color: #2D2D2D; border: 1px solid #444; padding: 5px; }
            QPushButton:hover { background-color: #3C3C3C; }
            QProgressBar { background-color: #2A2A2A; border: 1px solid #444; text-align: center; color: white; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #2A2A2A; border: 1px solid #444; padding: 5px; }
            QTabBar::tab:selected { background: #444; }
            QComboBox, QSpinBox, QSlider, QTextEdit { background-color: #1E1E1E; color: white; border: 1px solid #444; }
            QLabel { color: #CCCCCC; }
        """)
    window = ForexDashboard()
    window.show()
    sys.exit(app.exec_())
