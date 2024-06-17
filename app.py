import streamlit as st # type: ignore
import pandas as pd  # type: ignore
import numpy as np # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

global final_output
final_output = 0

st.write("""# House Price Predictor""")

df = pd.read_csv(r"kc_house_data.csv")
df.drop(columns = ['id','date','view','zipcode'],axis = 1,inplace = True)

estimators =  ['grade','sqft_living','bathrooms','yr_built','yr_renovated','bedrooms','sqft_living15','waterfront','sqft_lot','condition','floors','lat']

target = np.log(df['price'])
inputs = df[estimators]

inputs = df[estimators]
from sklearn.preprocessing import StandardScaler # type: ignore
scaler = StandardScaler()

scaler.fit(inputs)
scaled_inputs = scaler.transform(inputs)

x_train, x_test , y_train , y_test = train_test_split(scaled_inputs,target,test_size = 0.2, random_state = 5)
reg = LinearRegression()
reg.fit(x_train,y_train)

score = reg.score(x_test,y_test)

st.write("""### Accuracy""")
st.write(round(score,2) * 100, "%")
 
if final_output == 0 :
    st.write("#### Enter Values Below To Predict The House Price")
else :
    st.write(final_output)

grade = st.selectbox('Grade',(0,1,2,3,4,5,6,7,8,9,10))
sqft = st.number_input("Insert Size Of Build-Area In SqFt")
bathrooms = st.selectbox('Bathrooms',(0,1,2,3,4,5))
yr_built = st.number_input("Year Built",value = 0)
yr_renovate = st.number_input("Year Renovated",value = 0)
bedrooms = st.selectbox('Bedrooms',(0,1,2,3,4,5))
total_area = st.number_input("Insert Size Of House In SqFt")
waterfront = st.selectbox('Sea Facing',(0,1))
sqft2 = sqft
condition = st.selectbox('Condition',(0,1,2,3,4,5,6,7,8,9,10))
floors = st.selectbox('Floors',(0,1,2,3,4,5,6,7,8,9,10))
lat = st.number_input("Latitude",value = 47.5112)

def regression() :
    input_data = {'grade' : grade,
              'sqft_living' : sqft,
              'bathrooms':bathrooms,
              'yr_built':yr_built,
              'yr_renovated' : yr_renovate,
              'bedrooms':bedrooms,
              'sqft_living15' : sqft2,
              'waterfront':waterfront,
              'sqft_lot':total_area,
              'condition':condition,
              'floors':floors,
              'lat':lat}

    predict_data = pd.DataFrame(input_data,index=[0])
    scaled_inputs = scaler.transform(predict_data)

    prediction = reg.predict(scaled_inputs)
    prediction = np.exp(prediction)
    return prediction

if st.button("Predict"):
    final_output =  regression()
    st.write("""## House Price""")
    st.write("""##""",round(final_output[0],2))