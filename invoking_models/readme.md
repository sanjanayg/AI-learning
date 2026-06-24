run-backend end
cd .\invoking_models\src\
uvicorn main:app --reload 

front end
cd invoking_models
streamlit run app.py