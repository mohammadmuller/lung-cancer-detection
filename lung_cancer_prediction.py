# ----------------------------------
# بخش: کتابخانه های مورد نیاز
# ----------------------------------
import sys
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


# ----------------------------------
# بخش: کلاس آموزش مدل در ترد جداگانه
# ----------------------------------

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, df):
        super().__init__()
        self.df = df

    def run(self):
        try:
            self.progress.emit(10)

            cols_to_drop = ['Patient Id', 'Level'] if 'Level' in self.df.columns else ['Patient Id']
            cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]
            if cols_to_drop:
                self.df = self.df.drop(cols_to_drop, axis=1)


            X = self.df.drop('Result', axis=1)
            y = self.df['Result']


            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')


            y = y.astype(int)


            for col in X.columns:
                if X[col].isna().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        median_val = 5
                    X[col] = X[col].fillna(median_val)

            feature_names = X.columns.tolist()

            self.progress.emit(30)


            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            self.progress.emit(40)


            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            self.progress.emit(50)


            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1


            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'Logistic Regression': LogisticRegression(
                    C=0.1,
                    max_iter=5000,
                    class_weight='balanced',
                    random_state=42,
                    solver='liblinear'
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss',
                    random_state=42,
                    use_label_encoder=False,
                    verbosity=0
                ),
                'LightGBM': LGBMClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    class_weight='balanced',
                    random_state=42,
                    verbose=-1,
                    force_row_wise=True
                )
            }

            trained_models = {}
            scalers = {}
            accuracies = {}

            for i, (name, model) in enumerate(models.items()):
                self.progress.emit(50 + int((i + 1) * 40 / len(models)))

                # Train model
                model.fit(X_train, y_train)
                trained_models[name] = model
                scalers[name] = scaler

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies[name] = accuracy

                print(f"{name} accuracy: {accuracy:.4f}")

            self.progress.emit(100)
            self.finished.emit({
                'models': trained_models,
                'scalers': scalers,
                'feature_names': feature_names,
                'accuracies': accuracies
            })

        except Exception as e:
            self.error.emit(str(e))


# ----------------------------------
# بخش: کلاس اصلی برنامه
# ----------------------------------

class CancerRiskPredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.trained_models = None
        self.scalers = None
        self.feature_names = None
        self.accuracies = None
        self.df = None
        self.initUI()


        self.feature_mapping = {
            'سن': 'Age',
            'جنسیت': 'Gender',
            'آلودگی هوا': 'Air Pollution',
            'مصرف الکل': 'Alcohol use',
            'آلرژی به گرد و غبار': 'Dust Allergy',
            'مخاطرات شغلی': 'OccuPational Hazards',
            'ریسک ژنتیکی': 'Genetic Risk',
            'بیماری مزمن ریه': 'chronic Lung Disease',
            'رژیم غذایی متعادل': 'Balanced Diet',
            'چاقی': 'Obesity',
            'سیگار کشیدن': 'Smoking',
            'سیگاری غیرفعال': 'Passive Smoker',
            'درد قفسه سینه': 'Chest Pain',
            'سرفه خونی': 'Coughing of Blood',
            'خستگی': 'Fatigue',
            'کاهش وزن': 'Weight Loss',
            'تنگی نفس': 'Shortness of Breath',
            'خس خس سینه': 'Wheezing',
            'مشکل در بلع': 'Swallowing Difficulty',
            'تغییر شکل ناخن': 'Clubbing of Finger Nails',
            'سرماخوردگی مکرر': 'Frequent Cold',
            'سرفه خشک': 'Dry Cough',
            'خرخر کردن': 'Snoring'
        }

        self.persian_reverse_map = {v: k for k, v in self.feature_mapping.items()}

    # ----------------------------------
    # بخش: راه‌اندازی رابط کاربری
    # ----------------------------------

    def initUI(self):
        self.setWindowTitle('تشخیص خطر سرطان ریه')
        self.setGeometry(100, 100, 1500, 850)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #333333;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                font-family: monospace;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        prediction_tab = QWidget()
        tabs.addTab(prediction_tab, " پیش‌بینی خطر سرطان")

        training_tab = QWidget()
        tabs.addTab(training_tab, " آموزش مدل‌ها")

        results_tab = QWidget()
        tabs.addTab(results_tab, " تاریخچه نتایج")

        self.setupPredictionTab(prediction_tab)
        self.setupTrainingTab(training_tab)
        self.setupResultsTab(results_tab)

        self.prediction_history = []

    # ----------------------------------
    # بخش: تنظیم تب پیش‌بینی
    # ----------------------------------

    def setupPredictionTab(self, tab):
        layout = QHBoxLayout(tab)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        input_widget = QWidget()
        input_grid = QGridLayout(input_widget)
        input_grid.setColumnStretch(2, 1)

        self.feature_inputs = {}

        features = [
            ('سن', 'سن بر حسب سال', 0, 120, 45, 'سال'),
            ('جنسیت', 'جنسیت', ['مرد', 'زن'], None, 0, ''),
            ('آلودگی هوا', 'میزان قرارگیری در معرض آلودگی هوا (0=کمترین, 10=بیشترین)', 0, 10, 1, ''),
            ('مصرف الکل', 'میزان مصرف الکل (0=هیچ, 10=بسیار زیاد)', 0, 10, 1, ''),
            ('آلرژی به گرد و غبار', 'شدت آلرژی به گرد و غبار (0=بدون, 10=شدید)', 0, 10, 1, ''),
            ('مخاطرات شغلی', 'قرارگیری در معرض مخاطرات شغلی (0=بدون, 10=بسیار زیاد)', 0, 10, 1, ''),
            ('ریسک ژنتیکی', 'استعداد ژنتیکی به سرطان (0=کمترین, 10=بیشترین)', 0, 10, 1, ''),
            ('بیماری مزمن ریه', 'شدت بیماری مزمن ریه (0=بدون, 10=شدید)', 0, 10, 1, ''),
            ('رژیم غذایی متعادل', 'کیفیت رژیم غذایی (0=بسیار بد, 10=بسیار خوب)', 0, 10, 8, ''),
            ('چاقی', 'میزان چاقی (0=لاغر, 10=چاقی شدید)', 0, 10, 2, ''),
            ('سیگار کشیدن', 'میزان مصرف سیگار (0=هرگز, 10=بسیار زیاد)', 0, 10, 1, ''),
            ('سیگاری غیرفعال', 'قرارگیری در معرض دود سیگار (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('درد قفسه سینه', 'تکرار و شدت درد قفسه سینه (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('سرفه خونی', 'تکرار سرفه خونی (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('خستگی', 'میزان خستگی (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('کاهش وزن', 'کاهش وزن بی‌دلیل (0=هرگز, 10=شدید)', 0, 10, 1, ''),
            ('تنگی نفس', 'مشکل در تنفس (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('خس خس سینه', 'تکرار خس خس سینه (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('مشکل در بلع', 'مشکل در بلع غذا (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('تغییر شکل ناخن', 'تغییر شکل ناخن انگشتان (0=عادی, 10=شدید)', 0, 10, 1, ''),
            ('سرماخوردگی مکرر', 'تکرار سرماخوردگی (0=هرگز, 10=بسیار زیاد)', 0, 10, 1, ''),
            ('سرفه خشک', 'تکرار سرفه خشک (0=هرگز, 10=همیشه)', 0, 10, 1, ''),
            ('خرخر کردن', 'تکرار خرخر کردن در خواب (0=هرگز, 10=همیشه)', 0, 10, 1, '')
        ]

        row = 0
        for feature_data in features:
            feature_name = feature_data[0]
            description = feature_data[1]
            range_min = feature_data[2]
            range_max = feature_data[3]
            default = feature_data[4]

            label = QLabel(feature_name)
            label.setToolTip(description)
            label.setStyleSheet("font-weight: bold; min-width: 100px;")
            input_grid.addWidget(label, row, 0)

            if isinstance(range_min, list):
                combo = QComboBox()
                combo.addItems(range_min)
                combo.setCurrentIndex(default)
                self.feature_inputs[feature_name] = combo
                input_grid.addWidget(combo, row, 1)
            else:
                spinbox = QDoubleSpinBox()
                spinbox.setRange(range_min, range_max)
                spinbox.setValue(default)
                spinbox.setSingleStep(1)
                spinbox.setDecimals(0)
                self.feature_inputs[feature_name] = spinbox
                input_grid.addWidget(spinbox, row, 1)

            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #666666; font-size: 9px;")
            input_grid.addWidget(desc_label, row, 2)

            row += 1

        scroll.setWidget(input_widget)
        left_layout.addWidget(scroll)

        # Model selection and buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)

        model_group = QGroupBox("انتخاب مدل")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(['Random Forest', 'Logistic Regression', 'XGBoost', 'LightGBM'])
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        button_layout.addWidget(model_group)

        self.predict_btn = QPushButton(" پیش‌بینی خطر سرطان")
        self.predict_btn.clicked.connect(self.predict_risk)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setMinimumHeight(50)
        button_layout.addWidget(self.predict_btn)

        reset_btn = QPushButton(" تنظیم مجدد همه مقادیر")
        reset_btn.clicked.connect(self.reset_all_inputs)
        reset_btn.setMinimumHeight(50)
        button_layout.addWidget(reset_btn)

        left_layout.addWidget(button_widget)

        layout.addWidget(left_panel, 1)

        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        risk_group = QGroupBox(" ارزیابی خطر")
        risk_layout = QVBoxLayout()

        self.risk_label = QLabel("سطح خطر: محاسبه نشده")
        self.risk_label.setAlignment(Qt.AlignCenter)
        self.risk_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        risk_layout.addWidget(self.risk_label)

        self.risk_bar = QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_bar.setFormat("%p%")
        self.risk_bar.setMinimumHeight(30)
        risk_layout.addWidget(self.risk_bar)

        self.prob_label = QLabel("احتمال سرطان: --%")
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.prob_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        risk_layout.addWidget(self.prob_label)

        risk_group.setLayout(risk_layout)
        right_layout.addWidget(risk_group)

        details_group = QGroupBox(" تحلیل دقیق")
        details_layout = QVBoxLayout()
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(300)
        details_layout.addWidget(self.details_text)
        details_group.setLayout(details_layout)
        right_layout.addWidget(details_group)

        chart_group = QGroupBox(" نمایش گرافیکی خطر")
        chart_layout = QVBoxLayout()
        self.figure = Figure(figsize=(5, 2.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        chart_group.setLayout(chart_layout)
        right_layout.addWidget(chart_group)

        layout.addWidget(right_panel, 1)

    def reset_all_inputs(self):
        """Reset all input fields to default low-risk values"""
        for name, widget in self.feature_inputs.items():
            if name == 'سن':
                widget.setValue(45)
            elif name == 'جنسیت':
                widget.setCurrentIndex(0)
            elif name == 'رژیم غذایی متعادل':
                widget.setValue(8)
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(1)

        self.risk_label.setText("سطح خطر: محاسبه نشده")
        self.risk_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        self.risk_bar.setValue(0)
        self.prob_label.setText("احتمال سرطان: --%")
        self.details_text.clear()
        self.figure.clear()
        self.canvas.draw()

    # ----------------------------------
    # بخش: تنظیم تب آموزش
    # ----------------------------------

    def setupTrainingTab(self, tab):
        layout = QVBoxLayout(tab)

        info_label = QLabel(" لطفاً فایل دیتاست را بارگذاری کرده و مدل‌ها را آموزش دهید")
        info_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #e8f5e9; border-radius: 5px;")
        layout.addWidget(info_label)

        file_group = QGroupBox(" دیتاست")
        file_layout = QHBoxLayout()

        self.file_label = QLabel("هیچ فایلی بارگذاری نشده")
        self.file_label.setStyleSheet("padding: 8px; background-color: #f0f0f0; border-radius: 5px;")
        file_layout.addWidget(self.file_label)

        self.load_btn = QPushButton(" بارگذاری دیتاست")
        self.load_btn.clicked.connect(self.load_dataset)
        file_layout.addWidget(self.load_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        progress_group = QGroupBox(" پیشرفت آموزش")
        progress_layout = QVBoxLayout()

        self.training_progress = QProgressBar()
        progress_layout.addWidget(self.training_progress)

        self.status_label = QLabel(" آماده برای آموزش")
        self.status_label.setStyleSheet("padding: 5px;")
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        self.train_btn = QPushButton(" شروع آموزش مدل‌ها")
        self.train_btn.clicked.connect(self.train_models)
        self.train_btn.setEnabled(False)
        self.train_btn.setMinimumHeight(40)
        layout.addWidget(self.train_btn)

        info_group = QGroupBox(" اطلاعات دیتاست")
        info_layout = QVBoxLayout()
        self.dataset_info = QTextEdit()
        self.dataset_info.setReadOnly(True)
        self.dataset_info.setMaximumHeight(250)
        info_layout.addWidget(self.dataset_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

    # ----------------------------------
    # بخش: تنظیم تب تاریخچه
    # ----------------------------------

    def setupResultsTab(self, tab):
        layout = QVBoxLayout(tab)

        history_group = QGroupBox(" تاریخچه پیش‌بینی‌ها")
        history_layout = QVBoxLayout()

        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setFont(QFont("Courier", 10))
        history_layout.addWidget(self.history_list)

        clear_btn = QPushButton(" پاک کردن تاریخچه")
        clear_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(clear_btn)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

    # ----------------------------------
    # بخش: توابع بارگذاری و آموزش
    # ----------------------------------

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "انتخاب دیتاست", "", "CSV Files (*.csv)")

        if file_path:
            try:
                self.df = pd.read_csv(file_path)


                self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
                self.df = self.df.dropna(axis=1, how='all')


                self.df = self.df.loc[:, ~self.df.columns.str.contains('^\\.\\.\\.')]

                if 'Patient Id' in self.df.columns:
                    self.df = self.df.drop('Patient Id', axis=1)

                if 'Level' in self.df.columns:
                    self.df = self.df.drop('Level', axis=1)

                if 'Result' not in self.df.columns:
                    raise ValueError("دیتاست باید شامل ستون 'Result' باشد")

                for col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

                self.df = self.df.dropna(subset=['Result'])
                self.df['Result'] = self.df['Result'].astype(int)


                for col in self.df.columns:
                    if col != 'Result' and self.df[col].isna().any():
                        median_val = self.df[col].median()
                        if pd.isna(median_val):
                            median_val = 5
                        self.df[col] = self.df[col].fillna(median_val)

                self.file_label.setText(f" بارگذاری شد: {file_path.split('/')[-1]} ({len(self.df)} نمونه)")
                self.train_btn.setEnabled(True)

                feature_cols = [col for col in self.df.columns if col != 'Result']
                result_counts = self.df['Result'].value_counts()

                info_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                      اطلاعات دیتاست                         ║
╚══════════════════════════════════════════════════════════════╝

 تعداد نمونه‌ها: {len(self.df)}
 تعداد ویژگی‌ها: {len(feature_cols)}

 توزیع هدف:
   • بدون سرطان (0): {result_counts.get(0, 0)} نمونه ({result_counts.get(0, 0) / len(self.df) * 100:.1f}%)
   • با سرطان (1): {result_counts.get(1, 0)} نمونه ({result_counts.get(1, 0) / len(self.df) * 100:.1f}%)

 ویژگی‌ها ({len(feature_cols)} مورد):
   {', '.join(feature_cols[:20])}{'...' if len(feature_cols) > 20 else ''}

 داده‌ها آماده آموزش هستند.
                """
                self.dataset_info.setText(info_text)

            except Exception as e:
                QMessageBox.critical(self, "خطا", f"بارگذاری دیتاست با خطا مواجه شد:\n{str(e)}")
                self.file_label.setText(" بارگذاری ناموفق")
                self.train_btn.setEnabled(False)
                self.df = None

    def train_models(self):
        if self.df is None:
            QMessageBox.warning(self, "اخطار", "لطفاً ابتدا دیتاست را بارگذاری کنید!")
            return

        self.train_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.status_label.setText(" در حال آموزش مدل‌ها...")
        self.training_progress.setValue(0)

        self.training_thread = TrainingThread(self.df.copy())
        self.training_thread.progress.connect(self.training_progress.setValue)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.error.connect(self.on_training_error)
        self.training_thread.start()

    def on_training_finished(self, result):
        self.trained_models = result['models']
        self.scalers = result['scalers']
        self.feature_names = result['feature_names']
        self.accuracies = result.get('accuracies', {})

        best_model = max(self.accuracies, key=self.accuracies.get)
        best_acc = self.accuracies[best_model]

        acc_text = "\n".join([f"   • {name}: {acc:.2%}" for name, acc in self.accuracies.items()])
        self.status_label.setText(f" آموزش با موفقیت انجام شد! (بهترین: {best_model} - {best_acc:.2%})")
        self.predict_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.load_btn.setEnabled(True)

        QMessageBox.information(self, "موفقیت",
                                f" مدل‌ها با موفقیت آموزش دیدند!\n\n"
                                f"دقت مدل‌ها:\n{acc_text}\n\n"
                                f"اکنون می‌توانید پیش‌بینی را انجام دهید.")

    def on_training_error(self, error_msg):
        self.status_label.setText(" آموزش با خطا مواجه شد!")
        self.train_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "خطا", f"آموزش با خطا مواجه شد:\n{error_msg}")

    # ----------------------------------
    # بخش: تابع پیش‌بینی اصلی
    # ----------------------------------

    def predict_risk(self):
        if not self.trained_models:
            QMessageBox.warning(self, "اخطار", "لطفاً ابتدا مدل‌ها را آموزش دهید!")
            return

        try:
            # Collect input data
            input_data = {}
            for persian_name, widget in self.feature_inputs.items():
                english_name = self.feature_mapping.get(persian_name, persian_name)
                if isinstance(widget, QComboBox):
                    value = 0 if widget.currentText() == 'مرد' else 1
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    value = widget.value()
                else:
                    value = float(widget.text()) if widget.text() else 0
                input_data[english_name] = value

            # ============================================================
            # پیشبینی هیبریدی
            # ============================================================

            risk_weights = {
                'Smoking': 15,
                'Air Pollution': 10,
                'Genetic Risk': 12,
                'chronic Lung Disease': 10,
                'Chest Pain': 8,
                'Coughing of Blood': 12,
                'Shortness of Breath': 8,
                'Wheezing': 7,
                'Fatigue': 5,
                'Weight Loss': 8,
                'Swallowing Difficulty': 7,
                'Clubbing of Finger Nails': 8,
                'Obesity': 6,
                'Alcohol use': 5,
                'Dust Allergy': 4,
                'OccuPational Hazards': 6,
                'Passive Smoker': 5,
                'Frequent Cold': 4,
                'Dry Cough': 5,
                'Snoring': 3
            }


            protective_weights = {
                'Balanced Diet': 8
            }


            total_risk_weight = 0
            weighted_risk_sum = 0

            for feature, value in input_data.items():
                if feature == 'Age':

                    if value > 70:
                        age_risk = 12
                    elif value > 60:
                        age_risk = 8
                    elif value > 50:
                        age_risk = 5
                    else:
                        age_risk = 0
                    weighted_risk_sum += age_risk
                    total_risk_weight += 12

                elif feature == 'Gender':
                    pass

                elif feature in risk_weights:
                    weight = risk_weights[feature]

                    contribution = (value / 10.0) * weight
                    weighted_risk_sum += contribution
                    total_risk_weight += weight

                elif feature in protective_weights:

                    weight = protective_weights[feature]

                    risk_contribution = ((10 - value) / 10.0) * weight
                    weighted_risk_sum += risk_contribution
                    total_risk_weight += weight


            if total_risk_weight > 0:
                rule_based_risk = (weighted_risk_sum / total_risk_weight) * 100
            else:
                rule_based_risk = 0


            if rule_based_risk < 20:

                rule_based_risk = rule_based_risk * 0.5
            elif rule_based_risk > 80:

                rule_based_risk = min(98, rule_based_risk * 1.05)



            selected_model = self.model_combo.currentText()
            model = self.trained_models[selected_model]
            scaler = self.scalers[selected_model]


            input_df = pd.DataFrame([input_data])


            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 1

            input_df = input_df[self.feature_names]


            input_scaled = scaler.transform(input_df)


            if hasattr(model, 'predict_proba'):
                model_probability = model.predict_proba(input_scaled)[0][1]
            else:
                try:
                    decision = model.decision_function(input_scaled)
                    if len(decision.shape) > 1:
                        decision = decision[0]
                    model_probability = 1 / (1 + np.exp(-decision))
                    model_probability = max(0, min(1, model_probability))
                except:
                    model_probability = 0.5

            model_based_risk = model_probability * 100

            #مدل ترکیبی، برای دقت بیشتر
            if rule_based_risk < 15 or rule_based_risk > 85:

                final_risk = rule_based_risk * 0.8 + model_based_risk * 0.2
            elif rule_based_risk < 30 or rule_based_risk > 70:

                final_risk = rule_based_risk * 0.5 + model_based_risk * 0.5
            else:

                final_risk = rule_based_risk * 0.3 + model_based_risk * 0.7


            final_risk = max(0, min(100, final_risk))

            low_risk_features = ['Smoking', 'Air Pollution', 'Genetic Risk', 'chronic Lung Disease',
                                 'Chest Pain', 'Coughing of Blood', 'Alcohol use', 'Obesity']
            low_count = 0
            for feature in low_risk_features:
                if feature in input_data and input_data[feature] <= 1:
                    low_count += 1

            if low_count >= 6 and input_data.get('Age', 45) <= 50:
                final_risk = min(final_risk, 10)

            risk_percentage = final_risk

            if risk_percentage < 20:
                risk_level = "خطر بسیار پایین"
                risk_color = "#4CAF50"
                recommendation = " وضعیت عالی است! به سبک زندگی سالم خود ادامه دهید. معاینات سالانه را فراموش نکنید."
            elif risk_percentage < 35:
                risk_level = "خطر پایین"
                risk_color = "#8BC34A"
                recommendation = " خطر ابتلا پایین است. رعایت سبک زندگی سالم و چکاپ‌های منظم توصیه می‌شود."
            elif risk_percentage < 55:
                risk_level = "خطر متوسط"
                risk_color = "#FFC107"
                recommendation = " نیاز به توجه دارد. برای کاهش عوامل خطر با پزشک مشورت کنید. ترک سیگار و بهبود رژیم غذایی توصیه می‌شود."
            elif risk_percentage < 75:
                risk_level = "خطر بالا"
                risk_color = "#FF9800"
                recommendation = " خطر قابل توجه است. معاینه کامل پزشکی در اسرع وقت توصیه می‌شود. انجام سی‌تی اسکن ریه را بررسی کنید."
            else:
                risk_level = "خطر بسیار بالا"
                risk_color = "#F44336"
                recommendation = " وضعیت اورژانسی! فوراً به پزشک متخصص مراجعه کنید. تشخیص زودهنگام می‌تواند حیاتی باشد."

            self.risk_label.setText(f"سطح خطر: {risk_level}")
            self.risk_label.setStyleSheet(f"font-size: 20px; font-weight: bold; padding: 10px; color: {risk_color};")
            self.risk_bar.setValue(int(risk_percentage))

            if risk_percentage < 10:
                prob_display = f"احتمال سرطان: {risk_percentage:.1f}% (بسیار پایین)"
            elif risk_percentage < 30:
                prob_display = f"احتمال سرطان: {risk_percentage:.1f}% (پایین)"
            elif risk_percentage < 55:
                prob_display = f"احتمال سرطان: {risk_percentage:.1f}% (متوسط)"
            else:
                prob_display = f"احتمال سرطان: {risk_percentage:.1f}% (بالا)"

            self.prob_label.setText(prob_display)
            self.prob_label.setStyleSheet(f"font-size: 18px; font-weight: bold; padding: 10px; color: {risk_color};")


            self.figure.clear()
            ax = self.figure.add_subplot(111)


            zone_colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
            zone_bounds = [0, 20, 35, 55, 75, 100]

            for i in range(len(zone_bounds) - 1):
                ax.axvspan(zone_bounds[i], zone_bounds[i + 1], alpha=0.2, color=zone_colors[i])


            ax.barh([0], [risk_percentage], color='#2196F3', height=0.5, alpha=0.8, edgecolor='darkblue', linewidth=2)
            ax.axvline(x=risk_percentage, color='red', linestyle='--', linewidth=2, alpha=0.7)

            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('درصد خطر سرطان', fontsize=11, fontweight='bold')
            ax.set_title(' میزان خطر سرطان ریه', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

            zone_labels = ['کم خطر', 'پایین', 'متوسط', 'بالا', 'بسیار بالا']
            for i, label in enumerate(zone_labels):
                mid = (zone_bounds[i] + zone_bounds[i + 1]) / 2
                ax.text(mid, -0.35, label, fontsize=8, ha='center', color=zone_colors[i], fontweight='bold')

            self.canvas.draw()

            smoking = input_data.get('Smoking', 0)
            air_pollution = input_data.get('Air Pollution', 0)
            genetic_risk = input_data.get('Genetic Risk', 0)
            age = input_data.get('Age', 45)
            balanced_diet = input_data.get('Balanced Diet', 8)
            chest_pain = input_data.get('Chest Pain', 0)
            coughing_blood = input_data.get('Coughing of Blood', 0)

            details = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                            گزارش پیش‌بینی سرطان ریه                           ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │     اطلاعات مدل                                                               │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │   مدل استفاده شده: {selected_model}
    │   دقت مدل: {self.accuracies.get(selected_model, 0):.1%}
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │     نتیجه پیش‌بینی نهایی                                                      │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │   ╔══════════════════════════════════════════════════════════════════════╗
    │   ║   خطر تخمینی: {risk_percentage:.1f}%                                     ║
    │   ║   سطح خطر: {risk_level}                                                  ║
    │   ╚══════════════════════════════════════════════════════════════════════╝
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │     اطلاعات ورودی کلیدی                                                       │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │   • سن: {int(age)} سال
    │   • جنسیت: {'مرد' if input_data.get('Gender', 0) == 0 else 'زن'}
    │   • مصرف سیگار: {smoking}/10
    │   • آلودگی هوا: {air_pollution}/10
    │   • ریسک ژنتیکی: {genetic_risk}/10
    │   • رژیم غذایی: {balanced_diet}/10
    │   • درد قفسه سینه: {chest_pain}/10
    │   • سرفه خونی: {coughing_blood}/10
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │     تحلیل عوامل خطر                                                          │
    ├──────────────────────────────────────────────────────────────────────────────┤"""


            if age > 60:
                details += f"\n│    سن بالا: {int(age)} سال (+{min(15, (age - 50) // 2)}% خطر)"
            elif age > 50:
                details += f"\n│    سن: {int(age)} سال (+{min(8, (age - 50) // 3)}% خطر)"

            if smoking >= 8:
                details += f"\n│    مصرف سیگار: {smoking}/10 (خطر بسیار بالا)"
            elif smoking >= 5:
                details += f"\n│    مصرف سیگار: {smoking}/10 (خطر متوسط)"
            elif smoking >= 2:
                details += f"\n│    مصرف سیگار: {smoking}/10 (خطر کم)"
            else:
                details += f"\n│    مصرف سیگار: {smoking}/10 (خطر بسیار کم)"

            if air_pollution >= 8:
                details += f"\n│    آلودگی هوا: {air_pollution}/10 (خطر بسیار بالا)"
            elif air_pollution >= 5:
                details += f"\n│   آلودگی هوا: {air_pollution}/10 (خطر متوسط)"
            else:
                details += f"\n│    آلودگی هوا: {air_pollution}/10 (خطر کم)"

            if genetic_risk >= 8:
                details += f"\n│    ریسک ژنتیکی: {genetic_risk}/10 (خطر بسیار بالا)"
            elif genetic_risk >= 5:
                details += f"\n│    ریسک ژنتیکی: {genetic_risk}/10 (خطر متوسط)"
            else:
                details += f"\n│    ریسک ژنتیکی: {genetic_risk}/10 (خطر کم)"

            if balanced_diet <= 3:
                details += f"\n│    رژیم غذایی نامتعادل: {balanced_diet}/10 (خطر افزایش یافته)"
            elif balanced_diet <= 6:
                details += f"\n│    رژیم غذایی: {balanced_diet}/10 (نیاز به بهبود)"
            else:
                details += f"\n│    رژیم غذایی: {balanced_diet}/10 (خوب و محافظت کننده)"

            if chest_pain >= 7:
                details += f"\n│    درد قفسه سینه: {chest_pain}/10 (شدید - نیاز به بررسی فوری)"
            elif chest_pain >= 4:
                details += f"\n│    درد قفسه سینه: {chest_pain}/10 (متوسط)"

            if coughing_blood >= 5:
                details += f"\n│    سرفه خونی: {coughing_blood}/10 (علامت هشدار جدی)"

            details += f"""
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │    توصیه پزشکی اختصاصی                                                       │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │   {recommendation}
    └──────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │    اقدامات پیشنهادی                                                          │
    ├──────────────────────────────────────────────────────────────────────────────┤"""

            # Personalized recommendations
            if smoking > 0:
                details += f"\n│    ترک سیگار: مهم‌ترین اقدام برای کاهش خطر"
            if balanced_diet < 6:
                details += f"\n│    بهبود رژیم غذایی: مصرف میوه و سبزیجات بیشتر"
            if air_pollution > 4:
                details += f"\n│     استفاده از ماسک در روزهای آلوده"
            if risk_percentage > 40:
                details += f"\n│     انجام سی‌تی اسکن ریه با دوز پایین"

            details += f"""
    │ چکاپ سالانه و مشورت با پزشک متخصص ریه
    └──────────────────────────────────────────────────────────────────────────────┘

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║      توجه: این ابزار فقط برای پیش‌بینی است و جایگزین تشخیص پزشکی نمی‌شود.      ║
    ║      در صورت مشاهده هرگونه علائم، حتماً به پزشک مراجعه کنید.                   ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

            self.details_text.setText(details)


            from datetime import datetime
            history_entry = f"""
    ┌────────────────────────────────────────────────────────────────┐
    │  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    │  مدل: {selected_model}
    │  خطر: {risk_percentage:.1f}% ({risk_level})
    │  سن: {int(age)} | سیگار: {smoking}/10 | آلودگی: {air_pollution}/10
    └────────────────────────────────────────────────────────────────┘
    """
            self.history_list.append(history_entry)

            self.prediction_history.append({
                'timestamp': datetime.now(),
                'model': selected_model,
                'risk_percentage': risk_percentage,
                'risk_level': risk_level,
                'input_data': input_data
            })

        except Exception as e:
            QMessageBox.critical(self, "خطا", f"پیش‌بینی با خطا مواجه شد:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # ----------------------------------
    # بخش: توابع کمکی
    # ----------------------------------

    def clear_history(self):
        self.history_list.clear()
        self.prediction_history = []
        QMessageBox.information(self, "تاریخچه پاک شد", "تاریخچه پیش‌بینی‌ها با موفقیت پاک شد.")


# ----------------------------------
# بخش: اجرای اصلی برنامه
# ----------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = CancerRiskPredictor()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()