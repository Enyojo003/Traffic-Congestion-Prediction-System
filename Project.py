from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# Load and prepare data
traffic_data = pd.read_csv("./datasets/traffic_data.csv")

# Handling missing values
traffic_data.fillna(traffic_data.drop(columns=['timestamp']).mean(), inplace=True)

# Feature engineering
traffic_data['hour'] = pd.to_datetime(traffic_data['timestamp']).dt.hour
traffic_data['day_of_week'] = pd.to_datetime(traffic_data['timestamp']).dt.dayofweek

# Features and target
X = traffic_data[['hour', 'day_of_week']]
y = traffic_data['congestion']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building and training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Mapping day names to numeric values
day_mapping = {name: index for index, name in enumerate(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])}

# GUI Functions
def plot_congestion_prediction(hour, day_of_week):
    fig, ax = plt.subplots(figsize=(8, 6))
    hours = np.arange(0, 24)
    congestion = model.predict(scaler.transform(np.array([[h, day_of_week] for h in hours])))
    ax.plot(hours, congestion, label=f'Predicted Congestion for {list(day_mapping.keys())[day_of_week]}')
    ax.set_title('Predicted Traffic Congestion by Hour')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Predicted Congestion')
    ax.legend()
    return fig

def show_prediction():
    try:
        hour_str = hour_var.get()
        day_name = day_var.get()

        if 'AM' in hour_str:
            hour = int(hour_str.replace(' AM', ''))
            if hour == 12:
                hour = 0
        else:
            hour = int(hour_str.replace(' PM', ''))
            if hour != 12:
                hour += 12

        day_of_week = day_mapping.get(day_name)

        if hour < 0 or hour > 23 or day_of_week is None:
            messagebox.showerror("Input Error", "Please enter valid hour and day of the week.")
        else:
            congestion = predict_congestion(hour, day_of_week)
            result_label.config(text=f"Predicted Congestion: {congestion:.2f}")

            fig = plot_congestion_prediction(hour, day_of_week)
            for widget in plot_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.get_tk_widget().pack()
            canvas.draw()
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Input Error", "An error occurred while predicting congestion. Please check the input values.")

def predict_congestion(hour, day_of_week):
    new_data = np.array([[hour, day_of_week]])
    new_data_scaled = scaler.transform(new_data)
    predicted_congestion = model.predict(new_data_scaled)
    return predicted_congestion[0]

# GUI Setup
root = Tk()
root.title("Traffic Congestion Prediction System")

input_frame = Frame(root)
input_frame.pack(pady=20)

hour_var = StringVar()
hour_dropdown = ttk.Combobox(input_frame, textvariable=hour_var, values=[f"{i:02d} AM" if i < 12 else f"{i-12:02d} PM" for i in range(24)], state='readonly')
hour_dropdown.set("Select Hour")
hour_dropdown.grid(row=0, column=1, padx=10)

day_var = StringVar()
day_dropdown = ttk.Combobox(input_frame, textvariable=day_var, values=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], state='readonly')
day_dropdown.set("Select Day")
day_dropdown.grid(row=1, column=1, padx=10)

predict_button = Button(input_frame, text="Predict Congestion", command=show_prediction)
predict_button.grid(row=2, column=0, columnspan=2, pady=20)

result_label = Label(root, text="Predicted Congestion: N/A", font=("Helvetica", 16))
result_label.pack(pady=10)

plot_frame = Frame(root)
plot_frame.pack(pady=10)

root.mainloop()
