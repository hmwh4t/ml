import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timezone

# Cấu hình trang Streamlit: tiêu đề và layout
st.set_page_config(page_title="🌧️ Predict Rain App", layout="centered")
# Tiêu đề chính của ứng dụng
st.title("🌧️ Predict Rain (RainTomorrow)")

# Tải các mô hình và bộ mã hóa đã được lưu trữ từ trước
scaler = joblib.load("saved_models/scaler.joblib") # Tải đối tượng scaler để chuẩn hóa dữ liệu
pca = joblib.load("saved_models/pca_transformer.joblib") # Tải đối tượng PCA transformer để giảm chiều dữ liệu
rf_model = joblib.load("saved_models/random_forest_classifier_pca.joblib") # Tải mô hình Random Forest đã huấn luyện
dt_model = joblib.load("saved_models/decision_tree_classifier_pca.joblib") # Tải mô hình Decision Tree đã huấn luyện
label_encoders = joblib.load("saved_models/label_encoders.joblib") # Tải các bộ mã hóa nhãn (LabelEncoder)
rain_encoder = label_encoders.get("RainTomorrow", None) # Lấy cụ thể bộ mã hóa cho cột 'RainTomorrow'

# Tải thông tin độ chính xác của các mô hình đã lưu
accuracy_rf = joblib.load("saved_models/accuracy_rf.joblib") # Độ chính xác của mô hình Random Forest
accuracy_dt = joblib.load("saved_models/accuracy_dt.joblib") # Độ chính xác của mô hình Decision Tree

# Khóa API để truy cập dịch vụ WeatherAPI
API_KEY = '92f14b27fd924631b7c183633250106' # Thay thế bằng khóa API của bạn nếu cần

def convert_cloud_to_oktas(cloud_percent):
    """Chuyển đổi phần trăm độ che phủ của mây sang đơn vị oktas (thang đo từ 0 đến 8)."""
    if cloud_percent is None: # Nếu không có dữ liệu phần trăm mây
        return None
    if cloud_percent == 0: # Trời quang, không mây
        return 0
    elif cloud_percent <= 12.5: # Ít mây
        return 1
    elif cloud_percent <= 25:
        return 2
    elif cloud_percent <= 37.5:
        return 3
    elif cloud_percent <= 50: # Mây rải rác
        return 4
    elif cloud_percent <= 62.5:
        return 5
    elif cloud_percent <= 75: # Nhiều mây
        return 6
    elif cloud_percent <= 87.5:
        return 7
    else: # Trời u ám, mây che phủ hoàn toàn
        return 8

def get_simplified_weather_data(city_name, date):
    """Lấy dữ liệu thời tiết đơn giản hóa từ WeatherAPI cho một ngày cụ thể trong quá khứ."""
    weather_data = [city_name] # Khởi tạo danh sách chứa dữ liệu thời tiết, bắt đầu bằng tên thành phố
    
    try:
        base_url = "http://api.weatherapi.com/v1" # URL cơ sở của WeatherAPI
        
        # Sử dụng API lịch sử (history API) để lấy dữ liệu cho ngày đã qua
        history_url = f"{base_url}/history.json"
        history_params = {
            'key': API_KEY, # Khóa API
            'q': city_name, # Tên thành phố
            'dt': date,  # Ngày cần lấy dữ liệu (định dạng YYYY-MM-DD)
            'aqi': 'no' # Không yêu cầu dữ liệu chất lượng không khí (Air Quality Index)
        }
        
        # Gửi yêu cầu GET đến API lịch sử
        history_response = requests.get(history_url, params=history_params)
        history_response.raise_for_status() # Nếu có lỗi HTTP (ví dụ: 404, 500), sẽ raise exception
        history_data = history_response.json() # Chuyển đổi phản hồi JSON thành dictionary Python
        
        # Trích xuất dữ liệu thời tiết cho ngày cụ thể từ phản hồi
        day_data = history_data['forecast']['forecastday'][0] # Lấy dữ liệu của ngày đầu tiên trong forecastday (chỉ có 1 ngày)
        
        # Thêm các thông tin thời tiết vào danh sách weather_data
        weather_data.append(day_data['day']['mintemp_c'])  # Nhiệt độ tối thiểu (MinTemp)
        rainfall = day_data['day']['totalprecip_mm'] # Lượng mưa
        weather_data.append(rainfall)  # Lượng mưa (Rainfall)
        weather_data.append(1 if rainfall > 0 else 0)  # Có mưa hôm nay không (RainToday): 1 nếu có, 0 nếu không
        
        # Đối với dữ liệu lịch sử, sử dụng tốc độ gió tối đa trong ngày
        weather_data.append(day_data['day']['maxwind_kph'])  # Tốc độ gió giật mạnh nhất (WindGustSpeed)
        
        # Tìm dữ liệu thời tiết hàng giờ cho 9 giờ sáng và 3 giờ chiều
        hourly_data = day_data['hour'] # Danh sách dữ liệu thời tiết theo từng giờ
        weather_at_9am = None
        weather_at_3pm = None
        
        for hour_data in hourly_data:
            hour_time = datetime.fromisoformat(hour_data['time']).hour # Lấy giờ từ chuỗi thời gian ISO format
            if hour_time == 9: # Dữ liệu lúc 9 giờ sáng
                weather_at_9am = hour_data
            elif hour_time == 15:  # Dữ liệu lúc 3 giờ chiều (15h)
                weather_at_3pm = hour_data
        
        # Thêm các dữ liệu còn lại (nếu có)
        weather_data.append(weather_at_9am['wind_kph'] if weather_at_9am else None)  # Tốc độ gió lúc 9 giờ sáng (WindSpeed9am)
        weather_data.append(weather_at_3pm['wind_kph'] if weather_at_3pm else None)  # Tốc độ gió lúc 3 giờ chiều (WindSpeed3pm)
        weather_data.append(weather_at_9am['humidity'] if weather_at_9am else None)  # Độ ẩm lúc 9 giờ sáng (Humidity9am)
        weather_data.append(weather_at_3pm['humidity'] if weather_at_3pm else None)  # Độ ẩm lúc 3 giờ chiều (Humidity3pm)
        
        # Chuyển đổi phần trăm mây che phủ sang oktas
        cloud_9am_oktas = convert_cloud_to_oktas(weather_at_9am['cloud'] if weather_at_9am else None)
        cloud_3pm_oktas = convert_cloud_to_oktas(weather_at_3pm['cloud'] if weather_at_3pm else None)
        
        weather_data.append(cloud_9am_oktas)  # Mây lúc 9 giờ sáng (Cloud9am) tính bằng oktas
        weather_data.append(cloud_3pm_oktas)  # Mây lúc 3 giờ chiều (Cloud3pm) tính bằng oktas
            
    except requests.exceptions.RequestException as e: # Bắt lỗi liên quan đến mạng (không kết nối được, timeout, ...)
        st.error(f"Lỗi mạng: {e}")
        return None
    except KeyError as e: # Bắt lỗi khi truy cập key không tồn tại trong dictionary (thường do cấu trúc JSON thay đổi)
        st.error(f"Lỗi phân tích dữ liệu (KeyError): {e}")
        return None
    except Exception as e: # Bắt các lỗi không xác định khác
        st.error(f"Đã xảy ra lỗi khi lấy dữ liệu thời tiết cho {city_name}: {e}")
        return None
    
    return weather_data # Trả về danh sách dữ liệu thời tiết đã thu thập

def get_current_weather_data(city_name, date): # Mặc dù có tham số date, hàm này chủ yếu lấy dữ liệu hiện tại/dự báo hôm nay
    """Lấy dữ liệu thời tiết hiện tại và dự báo cho hôm nay từ WeatherAPI."""
    weather_data = [city_name] # Khởi tạo danh sách chứa dữ liệu thời tiết
    
    try:
        base_url = "http://api.weatherapi.com/v1" # URL cơ sở của WeatherAPI
        
        # API thời tiết hiện tại (current weather) để lấy tốc độ gió giật và một số dữ liệu dự phòng
        current_url = f"{base_url}/current.json"
        current_params = {
            'key': API_KEY,
            'q': city_name,
            'aqi': 'no' # Không yêu cầu dữ liệu chất lượng không khí
        }
        
        # API dự báo (forecast API) để lấy nhiệt độ tối thiểu hôm nay, lượng mưa, và dữ liệu hàng giờ
        forecast_url = f"{base_url}/forecast.json"
        forecast_params = {
            'key': API_KEY,
            'q': city_name,
            'days': 1, # Chỉ dự báo cho 1 ngày (hôm nay)
            'aqi': 'no', # Không yêu cầu dữ liệu chất lượng không khí
            'alerts': 'no' # Không yêu cầu thông tin cảnh báo thời tiết
        }
        
        # Lấy dữ liệu thời tiết hiện tại
        current_response = requests.get(current_url, params=current_params)
        current_response.raise_for_status()
        current_data = current_response.json()
        
        # Lấy dữ liệu dự báo
        forecast_response = requests.get(forecast_url, params=forecast_params)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        # Trích xuất dữ liệu dự báo cho ngày hôm nay
        today_forecast = forecast_data['forecast']['forecastday'][0]
        
        # Thêm các thông tin thời tiết vào danh sách
        weather_data.append(today_forecast['day']['mintemp_c'])  # Nhiệt độ tối thiểu (MinTemp)
        rainfall = today_forecast['day']['totalprecip_mm'] # Lượng mưa
        weather_data.append(rainfall)  # Lượng mưa (Rainfall)
        weather_data.append(1 if rainfall > 0 else 0)  # Có mưa hôm nay không (RainToday)
        weather_data.append(current_data['current']['gust_kph'])  # Tốc độ gió giật mạnh nhất (WindGustSpeed) từ dữ liệu hiện tại
        
        # Tìm dữ liệu thời tiết hàng giờ cho 9 giờ sáng và 3 giờ chiều từ dự báo
        hourly_data = today_forecast['hour']
        weather_at_9am = None
        weather_at_3pm = None
        
        for hour_data in hourly_data:
            hour_time = datetime.fromisoformat(hour_data['time']).hour
            if hour_time == 9:
                weather_at_9am = hour_data
            elif hour_time == 15: # 3pm
                weather_at_3pm = hour_data
        
        # Thêm các dữ liệu còn lại
        weather_data.append(weather_at_9am['wind_kph'] if weather_at_9am else None)  # Tốc độ gió lúc 9 giờ sáng (WindSpeed9am)
        weather_data.append(weather_at_3pm['wind_kph'] if weather_at_3pm else None)  # Tốc độ gió lúc 3 giờ chiều (WindSpeed3pm)
        weather_data.append(weather_at_9am['humidity'] if weather_at_9am else None)  # Độ ẩm lúc 9 giờ sáng (Humidity9am)
        weather_data.append(weather_at_3pm['humidity'] if weather_at_3pm else None)  # Độ ẩm lúc 3 giờ chiều (Humidity3pm)
        
        # Chuyển đổi phần trăm mây che phủ sang oktas
        cloud_9am_oktas = convert_cloud_to_oktas(weather_at_9am['cloud'] if weather_at_9am else None)
        cloud_3pm_oktas = convert_cloud_to_oktas(weather_at_3pm['cloud'] if weather_at_3pm else None)
        
        weather_data.append(cloud_9am_oktas)  # Mây lúc 9 giờ sáng (Cloud9am) tính bằng oktas
        weather_data.append(cloud_3pm_oktas)  # Mây lúc 3 giờ chiều (Cloud3pm) tính bằng oktas
            
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi mạng: {e}")
        return None
    except KeyError as e:
        st.error(f"Lỗi phân tích dữ liệu (KeyError): {e}")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy dữ liệu thời tiết cho {city_name}: {e}")
        return None
    
    return weather_data

# Các giá trị mặc định cho tất cả các cột đầu vào có thể có của mô hình
# Được sử dụng để điền vào các giá trị bị thiếu hoặc khi người dùng không cung cấp
default_values_full = {
    "Location": "Sydney", # Địa điểm mặc định
    "MinTemp": 12.0,      # Nhiệt độ tối thiểu mặc định
    "MaxTemp": 23.0,      # Nhiệt độ tối đa mặc định
    "Rainfall": 0.0,      # Lượng mưa mặc định
    "Evaporation": 3.2,   # Lượng bốc hơi mặc định
    "Sunshine": 9.8,      # Số giờ nắng mặc định
    "WindGustDir": "NW",  # Hướng gió giật mạnh nhất mặc định
    "WindGustSpeed": 39.0,# Tốc độ gió giật mạnh nhất mặc định
    "WindDir9am": "WNW",  # Hướng gió lúc 9h sáng mặc định
    "WindDir3pm": "WNW",  # Hướng gió lúc 3h chiều mặc định
    "WindSpeed9am": 13.0, # Tốc độ gió lúc 9h sáng mặc định
    "WindSpeed3pm": 19.0, # Tốc độ gió lúc 3h chiều mặc định
    "Humidity9am": 70.0,  # Độ ẩm lúc 9h sáng mặc định
    "Humidity3pm": 52.0,  # Độ ẩm lúc 3h chiều mặc định
    "Pressure9am": 1010.0,# Áp suất lúc 9h sáng mặc định
    "Pressure3pm": 1010.0,# Áp suất lúc 3h chiều mặc định
    "Cloud9am": 5,        # Độ che phủ mây lúc 9h sáng mặc định (oktas)
    "Cloud3pm": 5,        # Độ che phủ mây lúc 3h chiều mặc định (oktas)
    "Temp9am": 20.0,      # Nhiệt độ lúc 9h sáng mặc định
    "Temp3pm": 22.0,      # Nhiệt độ lúc 3h chiều mặc định
    "RainToday": 0        # Hôm nay có mưa không (0: Không, 1: Có)
}

# Danh sách 10 đặc trưng được chọn để đưa vào mô hình, theo đúng thứ tự đã huấn luyện
selected_features = [
    "Humidity3pm", "RainToday", "Cloud3pm", "Humidity9am", "Cloud9am",
    "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "MinTemp"
]

# === CHỌN CÁCH NHẬP DỮ LIỆU ===
# Tạo một radio button cho phép người dùng chọn phương thức nhập dữ liệu
input_mode = st.radio("📅 Bạn muốn nhập dữ liệu bằng cách nào?", 
                      ["Nhập thủ công", "Lấy từ WeatherAPI", "Tải lên tệp CSV"],
                      help="Chọn cách bạn muốn cung cấp dữ liệu thời tiết để dự đoán.")

def fill_missing_with_defaults(df):
    """Điền các giá trị bị thiếu (NaN) trong DataFrame bằng các giá trị mặc định đã định nghĩa."""
    for col, default_value in default_values_full.items(): # Lặp qua từng cột và giá trị mặc định của nó
        if col in df.columns: # Kiểm tra xem cột có tồn tại trong DataFrame không
            df[col] = df[col].fillna(default_value) # Điền giá trị NaN bằng giá trị mặc định
    return df # Trả về DataFrame đã được xử lý

def process_prediction(input_df, model_type):
    """Xử lý DataFrame đầu vào, thực hiện dự đoán và hiển thị kết quả."""
    # Xử lý cột 'RainToday': chuyển đổi 'Yes'/'No' thành 1/0 nếu cần
    if 'RainToday' in input_df.columns:
        # Ánh xạ 'Yes' thành 1, 'No' thành 0. Nếu giá trị đã là số hoặc NaN, giữ nguyên.
        input_df['RainToday'] = input_df['RainToday'].map({'Yes': 1, 'No': 0}).fillna(input_df['RainToday'])

    # Mã hóa các cột dạng object (categorical) bằng LabelEncoder đã lưu
    # Chỉ áp dụng cho các cột có trong `label_encoders` và có kiểu dữ liệu là object
    for col in input_df.columns:
        if col in label_encoders and input_df[col].dtype == object:
            # Chuyển đổi cột sang kiểu string trước khi transform để tránh lỗi
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

    # Chuẩn hóa dữ liệu bằng scaler đã tải
    X_scaled = scaler.transform(input_df)
    # Áp dụng phép biến đổi PCA
    X_pca = pca.transform(X_scaled)

    # Chọn mô hình và độ chính xác tương ứng dựa trên lựa chọn của người dùng
    if model_type == "Random Forest":
        model = rf_model
        accuracy = accuracy_rf
    else: # Decision Tree
        model = dt_model
        accuracy = accuracy_dt
        
    prediction = model.predict(X_pca)[0] # Dự đoán cho dòng dữ liệu đầu tiên (hoặc duy nhất)
    # Chuyển đổi kết quả dự đoán (0 hoặc 1) sang nhãn "No" hoặc "Yes"
    result_label = {0: "Không", 1: "Có"}.get(prediction, str(prediction)) # Mặc định là chuỗi của prediction nếu không phải 0 hoặc 1
    emoji = "☔" if prediction == 1 else "🌤️" # Chọn emoji tương ứng với kết quả

    # Hiển thị kết quả dự đoán
    st.success(f"🌟 Kết quả dự đoán thời tiết: **{emoji} {result_label}** (bởi mô hình {model_type})")
    st.info(f"📊 Độ chính xác của mô hình ({model_type}) trên dữ liệu kiểm tra: **{accuracy*100:.2f}%**")
    
    # Hiển thị xác suất dự đoán
    proba = model.predict_proba(X_pca)[0] # Lấy xác suất cho các lớp
    st.subheader("🧪 Xác suất có mưa vào ngày mai (RainTomorrow):")
    st.bar_chart({"Không mưa": proba[0], "Có mưa": proba[1]}) # Vẽ biểu đồ cột cho xác suất

# Logic xử lý dựa trên chế độ nhập liệu người dùng chọn
if input_mode == "Lấy từ WeatherAPI":
    # Tạo một form để nhóm các widget nhập liệu
    with st.form("weatherapi_form"):
        st.subheader("🌤️ Lấy dữ liệu thời tiết từ WeatherAPI:")
        city_name = st.text_input("Tên thành phố", "Ho Chi Minh", help="Nhập tên thành phố bạn muốn dự báo.")
        date = st.date_input("Ngày", datetime.now().date(), help="Chọn ngày bạn muốn lấy dữ liệu.")
        model_type = st.selectbox("🧠Chọn một mô hình", ["Random Forest", "Decision Tree"], help="Chọn mô hình để dự đoán.")
        submit = st.form_submit_button("Lấy dữ liệu & Dự đoán") # Nút để gửi form

    if submit and city_name: # Nếu nút được nhấn và tên thành phố được nhập
        date_str = date.strftime("%Y-%m-%d") # Định dạng ngày thành chuỗi YYYY-MM-DD
        
        # Kiểm tra nếu ngày được chọn là hôm nay hoặc tương lai, sử dụng API thời tiết hiện tại/dự báo.
        # Ngược lại (ngày trong quá khứ), sử dụng API lịch sử.
        if date >= datetime.now().date():
            with st.spinner(f"Đang lấy dữ liệu thời tiết hiện tại/dự báo cho {city_name}..."):
                weather_data = get_current_weather_data(city_name, date_str)
        else:
            with st.spinner(f"Đang lấy dữ liệu thời tiết lịch sử cho {city_name} vào ngày {date_str}..."):
                weather_data = get_simplified_weather_data(city_name, date_str)
            
        if weather_data: # Nếu lấy dữ liệu thành công
            # Tạo DataFrame từ dữ liệu lấy được, điền các giá trị còn thiếu bằng giá trị mặc định
            # Thứ tự các phần tử trong weather_data:
            # [location, mintemp, rainfall, raintoday, windgustspeed, windspeed9am, windspeed3pm, humidity9am, humidity3pm, cloud9am, cloud3pm]
            full_input_dict = {
                "Location": weather_data[0],
                "MinTemp": weather_data[1],
                "Rainfall": weather_data[2],
                "RainToday": weather_data[3],
                "WindGustSpeed": weather_data[4],
                "WindSpeed9am": weather_data[5],
                "WindSpeed3pm": weather_data[6],
                "Humidity9am": weather_data[7],
                "Humidity3pm": weather_data[8],
                "Cloud9am": weather_data[9],
                "Cloud3pm": weather_data[10],
            }
            # Tạo DataFrame một dòng từ dictionary, các cột không có trong full_input_dict sẽ là NaN ban đầu
            full_input = pd.DataFrame([full_input_dict])
            # Điền các giá trị mặc định cho các cột còn lại (MaxTemp, Evaporation, etc.)
            full_input = fill_missing_with_defaults(full_input)
            # Chọn các đặc trưng cần thiết cho mô hình
            input_df_for_prediction = full_input[selected_features].copy()
            
            st.subheader("📊 Dữ liệu thời tiết đã lấy (sau khi điền giá trị mặc định):")
            st.write(full_input) # Hiển thị DataFrame đầy đủ
            
            # Thực hiện dự đoán
            process_prediction(input_df_for_prediction, model_type)

elif input_mode == "Nhập thủ công":
    # Hàm tiện ích để chuyển đổi đầu vào văn bản thành float, trả về None nếu không hợp lệ
    def parse_float(value_str):
        try:
            return float(value_str)
        except ValueError: # Nếu không thể chuyển đổi sang float
            return None # Trả về None để sau này có thể điền giá trị mặc định

    with st.form("input_form"):
        st.subheader("🔢 Nhập dữ liệu dự báo thời tiết thủ công:")
        # Tạo các cột để bố trí gọn gàng hơn
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.text_input("Địa điểm (Location)", default_values_full["Location"])
            min_temp_str = st.text_input("Nhiệt độ tối thiểu - MinTemp (°C)", str(default_values_full["MinTemp"]))
            max_temp_str = st.text_input("Nhiệt độ tối đa - MaxTemp (°C)", str(default_values_full["MaxTemp"]))
            rainfall_str = st.text_input("Lượng mưa - Rainfall (mm)", str(default_values_full["Rainfall"]))
            evaporation_str = st.text_input("Lượng bốc hơi - Evaporation (mm)", str(default_values_full["Evaporation"]))
            sunshine_str = st.text_input("Số giờ nắng - Sunshine (hours)", str(default_values_full["Sunshine"]))
            wind_gust_dir = st.text_input("Hướng gió giật mạnh nhất - WindGustDir", default_values_full["WindGustDir"])
            wind_gust_speed_str = st.text_input("Tốc độ gió giật mạnh nhất - WindGustSpeed (km/h)", str(default_values_full["WindGustSpeed"]))
            temp_9am_str = st.text_input("Nhiệt độ lúc 9h sáng - Temp9am (°C)", str(default_values_full["Temp9am"]))
            
        with col2:
            wind_dir_9am = st.text_input("Hướng gió lúc 9h sáng - WindDir9am", default_values_full["WindDir9am"])
            wind_dir_3pm = st.text_input("Hướng gió lúc 3h chiều - WindDir3pm", default_values_full["WindDir3pm"])
            wind_speed_9am_str = st.text_input("Tốc độ gió lúc 9h sáng - WindSpeed9am (km/h)", str(default_values_full["WindSpeed9am"]))
            wind_speed_3pm_str = st.text_input("Tốc độ gió lúc 3h chiều - WindSpeed3pm (km/h)", str(default_values_full["WindSpeed3pm"]))
            humidity_9am_str = st.text_input("Độ ẩm lúc 9h sáng - Humidity9am (%)", str(default_values_full["Humidity9am"]))
            humidity_3pm_str = st.text_input("Độ ẩm lúc 3h chiều - Humidity3pm (%)", str(default_values_full["Humidity3pm"]))
            pressure_9am_str = st.text_input("Áp suất lúc 9h sáng - Pressure9am (hPa)", str(default_values_full["Pressure9am"]))
            pressure_3pm_str = st.text_input("Áp suất lúc 3h chiều - Pressure3pm (hPa)", str(default_values_full["Pressure3pm"]))
            temp_3pm_str = st.text_input("Nhiệt độ lúc 3h chiều - Temp3pm (°C)", str(default_values_full["Temp3pm"]))

        # Các trường nhập liệu không nằm trong cột
        cloud_9am_str = st.text_input("Mây lúc 9h sáng - Cloud9am (0-8 oktas)", str(default_values_full["Cloud9am"]))
        cloud_3pm_str = st.text_input("Mây lúc 3h chiều - Cloud3pm (0-8 oktas)", str(default_values_full["Cloud3pm"]))
        rain_today_options = ["No", "Yes"]
        rain_today_default_index = 0 if default_values_full["RainToday"] == 0 else 1
        rain_today = st.selectbox("Hôm nay có mưa không - RainToday", rain_today_options, index=rain_today_default_index)
        
        model_type = st.selectbox("🧠Chọn một mô hình", ["Random Forest", "Decision Tree"])
        submit = st.form_submit_button("Dự đoán")

    if submit:
        # Tạo dictionary từ dữ liệu người dùng nhập, chuyển đổi sang float nếu có thể
        manual_input_data = {
            "Location": location,
            "MinTemp": parse_float(min_temp_str),
            "MaxTemp": parse_float(max_temp_str),
            "Rainfall": parse_float(rainfall_str),
            "Evaporation": parse_float(evaporation_str),
            "Sunshine": parse_float(sunshine_str),
            "WindGustDir": wind_gust_dir,
            "WindGustSpeed": parse_float(wind_gust_speed_str),
            "WindDir9am": wind_dir_9am,
            "WindDir3pm": wind_dir_3pm,
            "WindSpeed9am": parse_float(wind_speed_9am_str),
            "WindSpeed3pm": parse_float(wind_speed_3pm_str),
            "Humidity9am": parse_float(humidity_9am_str),
            "Humidity3pm": parse_float(humidity_3pm_str),
            "Pressure9am": parse_float(pressure_9am_str),
            "Pressure3pm": parse_float(pressure_3pm_str),
            "Cloud9am": parse_float(cloud_9am_str),
            "Cloud3pm": parse_float(cloud_3pm_str),
            "Temp9am": parse_float(temp_9am_str),
            "Temp3pm": parse_float(temp_3pm_str),
            "RainToday": rain_today # Giữ nguyên là "Yes" hoặc "No" để hàm process_prediction xử lý
        }
        
        # Tạo DataFrame một dòng từ dictionary
        full_input = pd.DataFrame([manual_input_data])
        # Điền các giá trị None (do parse_float không thành công hoặc người dùng để trống) bằng giá trị mặc định
        full_input = fill_missing_with_defaults(full_input)
        # Chọn các đặc trưng cần thiết cho mô hình
        input_df_for_prediction = full_input[selected_features].copy()
        
        st.subheader("📊 Dữ liệu đã nhập (sau khi điền giá trị mặc định):")
        st.write(full_input)

        process_prediction(input_df_for_prediction, model_type)

else: # Chế độ: Tải lên tệp CSV
    uploaded_file = st.file_uploader("📁 Tải lên một tệp CSV chứa dữ liệu đầu vào", type=["csv"], help="Tệp CSV phải có các cột tương ứng với các đặc trưng thời tiết.")
    model_type = st.selectbox("🧠Chọn một mô hình", ["Random Forest", "Decision Tree"])

    if uploaded_file is not None: # Nếu người dùng đã tải tệp lên
        try:
            uploaded_df = pd.read_csv(uploaded_file) # Đọc tệp CSV vào DataFrame
            st.write("Xem trước dữ liệu đã tải lên (5 dòng đầu):")
            st.dataframe(uploaded_df.head())

            # Điền các giá trị thiếu trong DataFrame đã tải lên bằng giá trị mặc định
            processed_df = fill_missing_with_defaults(uploaded_df.copy()) # Làm việc trên bản sao
            
            # Chọn các đặc trưng cần thiết từ DataFrame đã tải lên
            input_df_for_prediction = processed_df[selected_features].copy()

            # Xử lý cột 'RainToday' nếu tồn tại và là dạng chuỗi
            if 'RainToday' in input_df_for_prediction.columns:
                input_df_for_prediction['RainToday'] = input_df_for_prediction['RainToday'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1:1, 0:0}).fillna(input_df_for_prediction['RainToday'])


            # Mã hóa các cột object (categorical)
            for col in input_df_for_prediction.columns:
                if col in label_encoders and input_df_for_prediction[col].dtype == object:
                    try:
                        input_df_for_prediction[col] = label_encoders[col].transform(input_df_for_prediction[col].astype(str))
                    except ValueError as ve:
                        st.error(f"Lỗi khi mã hóa cột '{col}': {ve}. Vui lòng kiểm tra các giá trị trong cột này. Các giá trị phải nằm trong tập đã huấn luyện encoder.")
                        st.stop() # Dừng thực thi nếu có lỗi mã hóa

            # Chuẩn hóa và áp dụng PCA
            X_scaled = scaler.transform(input_df_for_prediction)
            X_pca = pca.transform(X_scaled)

            # Chọn mô hình
            model = rf_model if model_type == "Random Forest" else dt_model
            
            # Thực hiện dự đoán cho tất cả các hàng trong DataFrame
            predictions = model.predict(X_pca)
            probabilities = model.predict_proba(X_pca) # Lấy xác suất

            # Tạo DataFrame kết quả
            output_df = uploaded_df.copy() # Bắt đầu với DataFrame gốc đã tải lên
            output_df["Dự đoán (RainTomorrow)"] = predictions # Thêm cột dự đoán (0 hoặc 1)
            output_df["Nhãn dự đoán"] = output_df["Dự đoán (RainTomorrow)"].map({0: "Không mưa", 1: "Có mưa"}) # Thêm cột nhãn
            output_df["Xác suất không mưa"] = probabilities[:, 0] # Xác suất của lớp 0
            output_df["Xác suất có mưa"] = probabilities[:, 1]   # Xác suất của lớp 1

            # Chuyển đổi DataFrame kết quả thành CSV để tải xuống
            csv_output = output_df.to_csv(index=False).encode("utf-8")
            st.success("✅ Dự đoán hoàn tất cho tệp đã tải lên. Bạn có thể tải xuống kết quả bên dưới:")
            st.download_button(
                label="📅 Tải xuống kết quả dự đoán (CSV)",
                data=csv_output,
                file_name="ket_qua_du_doan_mua.csv", # Tên tệp tải xuống
                mime="text/csv"
            )
            st.write("Xem trước kết quả dự đoán:")
            st.dataframe(output_df.head())

        except KeyError as e: # Xử lý lỗi nếu thiếu cột cần thiết trong tệp CSV
            st.error(f"❌ Lỗi: Tệp CSV bị thiếu cột bắt buộc: {e}. Vui lòng đảm bảo tệp CSV có đủ các cột: {', '.join(selected_features)}")
        except Exception as e: # Bắt các lỗi khác có thể xảy ra khi xử lý tệp
            st.error(f"Đã xảy ra lỗi khi xử lý tệp CSV: {e}")
