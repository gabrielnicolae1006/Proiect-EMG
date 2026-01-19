import numpy as np
import os
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class LoadData:
    def __init__(self, dataDirectory, dc_offset=128):
        self.dataDirectory = dataDirectory
        self.dc_offset = dc_offset

    def load_data(self, filename):
        return np.load(filename)

    def _get_channels(self, file_data, n_channels=None):
        if n_channels is None:
            n_channels = file_data.shape[0]
        channels = [
            file_data[i].astype(int) - self.dc_offset
            for i in range(n_channels)
        ]
        return channels

    def _iter_npy_files(self):
        if not os.path.exists(self.dataDirectory):
            print(f"Error: the folder '{self.dataDirectory}' doesn't exist")
            return

        for filename in os.listdir(self.dataDirectory):
            if filename.endswith(".npy"):
                full_path = os.path.join(self.dataDirectory, filename)
                file_data = self.load_data(full_path)
                parts = filename.split("_")
                yield filename, parts, file_data

    def loadData_armthreeClasses(self, n_channels=8, class_index_in_name=2):
        dataStore = []
        labels = []
        class_map = {"0": 0, "1": 1, "2": 2}

        for filename, parts, file_data in self._iter_npy_files():
            if len(parts) <= class_index_in_name:
                continue
            cl = parts[class_index_in_name]
            if cl in class_map:
                channels = self._get_channels(file_data, n_channels=n_channels)
                dataStore.append(channels)
                labels.append(class_map[cl])
        return dataStore, labels


class EMGFeatureExtractor:

    def compute_mav(self, x):
        return np.mean(np.abs(x))

    def compute_rms(self, x):
        return np.sqrt(np.mean(x ** 2))

    def compute_wl(self, x):
        return np.sum(np.abs(np.diff(x)))

    def compute_zcr(self, x, threshold=0.0):
        count = 0
        for k in range(1, len(x)):
            if (x[k] * x[k - 1] < 0) and (abs(x[k] - x[k - 1]) >= threshold):
                count += 1
        return count

    def compute_ssc(self, x, threshold=0.0):
        count = 0
        for k in range(1, len(x) - 1):
            diff1 = x[k] - x[k - 1]
            diff2 = x[k] - x[k + 1]
            if (diff1 * diff2 >= threshold):
                count += 1
        return count

    def compute_skewness(self, x):
        return skew(x) 

    def extract_features(self, window):
        feats = []
        feats.append(self.compute_mav(window))
        feats.append(self.compute_rms(window))
        feats.append(self.compute_wl(window))
        feats.append(self.compute_zcr(window, threshold=0.1))
        feats.append(self.compute_ssc(window, threshold=0.1))
        feats.append(self.compute_skewness(window))
        return feats



def process_emg_dataset(data_path, window_size=200, overlap=100):
    """
    Enter data, apply windowing and extract features.
    Return X (features) and y (labels).
    """
    print("1. Initializing LoadData...")
    loader = LoadData(data_path, dc_offset=128)

    print("2. Loading signal (Arm Three Classes)...")

    raw_data, raw_labels = loader.loadData_armthreeClasses()

    if len(raw_data) == 0:
        print("No data found")
        return None, None

    print(f"   ->  {len(raw_data)} loaded.")

    extractor = EMGFeatureExtractor()
    X_features = []
    y_labels = []

    print("3. Windowing + Feature Extraction")

    step = window_size - overlap


    for i in range(len(raw_data)):
        channels = raw_data[i] 
        label = raw_labels[i] 


        n_samples = len(channels[0])


        for start in range(0, n_samples - window_size, step):
            window_feature_vector = []

            for ch_idx in range(8):
                window = channels[ch_idx][start: start + window_size]


                ch_feats = extractor.extract_features(window)


                window_feature_vector.extend(ch_feats)


            X_features.append(window_feature_vector)
            y_labels.append(label)

    X = np.array(X_features)
    y = np.array(y_labels)

    print("COmplete")
    print(f"X (Features): {X.shape}")
    print(f"y (Labels): {y.shape}")

    return X, y



PATH_TO_DATA = r"."


if PATH_TO_DATA != "npy_path":
    X, y = process_emg_dataset(PATH_TO_DATA, window_size=200, overlap=50)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")


print("\nInitialization and training of Random Forest")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("Complete")


print("\nTest Evaluation:")
y_pred = clf.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print(f"--> Global Accuracy: {acc * 100:.2f}%")

print("\n Detailed report of classification")

target_names = ['Exercise 0', 'Exercise 1', 'Exercise 2']
print(classification_report(y_test, y_pred, target_names=target_names))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predict')
plt.ylabel('Reality')
plt.title('Confusion Matrix - Movement Classification')
plt.show()