import logging
import os
import pandas as pd
import pynance as pn


class DataPreprocessor:
    """
    DataPreprocessor class for fetching, detecting, cleaning, and analyzing financial data from YFinance.
    """

    def __init__(
        self,
        data_dir: str = "../data",
        log_file: str = "../logs/data_preprocessing.log",
        log_level: int = logging.INFO,
    ):
        """
        Initializes the DataPreprocessor instance.

        Parameters:
        - data_dir (str): Directory to save downloaded data. Defaults to "../data".
        - log_file (str): Log file path. Defaults to "../logs/data_preprocessing.log".
        - log_level (int): Logging level. Defaults to logging.INFO.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger = self._setup_logging(log_file, log_level)

    def _setup_logging(self, log_file, log_level):
        """Sets up logging to prevent duplicate handlers."""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logger.setLevel(log_level)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter("%(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def get_data(self, start_date, end_date, symbols):
        """
        Fetches historical data for each symbol and saves it as a CSV.

        Returns:
        - dict: Dictionary with symbol names as keys and file paths of saved CSV files as values.
        """
        data_paths = {}

        for symbol in symbols:
            try:
                self.logger.info(
                    f"📊 Fetching data for {symbol} from {start_date} to {end_date}..."
                )
                data = pn.data.get(symbol, start=start_date, end=end_date)
                file_path = os.path.join(self.data_dir, f"{symbol}.csv")
                data.to_csv(file_path)

                normalized_path = file_path.replace("\\", "/")
                data_paths[symbol] = normalized_path
                self.logger.info(f"✅ Data for {symbol} saved to '{normalized_path}'.")

            except ValueError as ve:
                error_message = f"⚠️ Data format issue for {symbol}: {ve}"
                self.logger.error(error_message)

            except Exception as e:
                error_message = f"❌ Failed to fetch data for {symbol}: {e}"
                self.logger.error(error_message)

        return data_paths
