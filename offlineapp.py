import sys
import json
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QCheckBox, QSpinBox
from stock_prediction import stock_prediction
class StockPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input fields
        self.tickers_input = QLineEdit(self)
        self.tickers_input.setPlaceholderText("Enter tickers (comma-separated)")

        self.target_ticker_input = QLineEdit(self)
        self.target_ticker_input.setPlaceholderText("Enter target ticker")

        self.future_minutes_input = QSpinBox(self)
        self.future_minutes_input.setRange(1, 60)
        self.future_minutes_input.setValue(4)

        self.sentiment_checkbox = QCheckBox("Enable Sentiment Analysis", self)

        # Button to trigger prediction
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.get_prediction)

        # Output display
        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)

        # Add widgets to layout
        layout.addWidget(QLabel("Tickers:"))
        layout.addWidget(self.tickers_input)
        layout.addWidget(QLabel("Target Ticker:"))
        layout.addWidget(self.target_ticker_input)
        layout.addWidget(QLabel("Future Minutes:"))
        layout.addWidget(self.future_minutes_input)
        layout.addWidget(self.sentiment_checkbox)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_display)

        self.setLayout(layout)
        self.setWindowTitle("Stock Market Prediction")
        self.resize(500, 400)

    def get_prediction(self):
        # Get user input
        tickers = [ticker.strip() for ticker in self.tickers_input.text().split(",")]
        target_ticker = self.target_ticker_input.text().strip()
        num_future_minutes = self.future_minutes_input.value()
        sentiment_analysis = self.sentiment_checkbox.isChecked()

        if not tickers or not target_ticker:
            self.result_display.setPlainText("Please enter valid tickers and target ticker.")
            return  

        # Call the stock prediction function (Replace this with actual function call)
        prediction_json= stock_prediction(tickers, target_ticker, num_future_minutes, sentiment_analysis=sentiment_analysis, sentiment_interval= 24)
        prediction_output = json.loads(prediction_json)
        # Format the output
        formatted_output = f"📈 Stock Predictions for {target_ticker}\n\n"

        # Multi-stock Predictions
        formatted_output += "🔹 Recent Predictions:\n"
        for pred in prediction_output["multi_stock_predictions"]:
            formatted_output += (f"  - {pred['datetime']}\n"
                                 f"    Actual Close: ${pred['actual_close']:.2f}\n"
                                 f"    Predicted Close: ${pred['predicted_close']:.2f}\n")

        formatted_output += "\n🔮 Future Predictions:\n"

        # Future Predictions
        for pred in prediction_output["future_predictions"]:
            formatted_output += (f"  - In {pred['minute_offset']} min ({pred['predicted_time']}):\n"
                                 f"    Predicted Close: ${pred['predicted_close']:.2f}\n")

        # Display in UI
        self.result_display.setPlainText(formatted_output)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec())
