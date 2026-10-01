import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


st.set_page_config(page_title='Loan Risk Prediction', page_icon='💳', layout='wide')


@st.cache_data
def load_dataset():
    return pd.read_csv('loan_data.csv')


@st.cache_data
def train_model():
    loans = load_dataset()
    final_data = pd.get_dummies(loans, columns=['purpose'], drop_first=True)

    X = final_data.drop('not.fully.paid', axis=1)
    y = final_data['not.fully.paid']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=101
    )

    model = RandomForestClassifier(n_estimators=600, random_state=101)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, preds


@st.cache_data
def get_model_columns():
    loans = load_dataset()
    final_data = pd.get_dummies(loans, columns=['purpose'], drop_first=True)
    return final_data.drop('not.fully.paid', axis=1).columns.tolist()


def build_single_record(user_input):
    columns = get_model_columns()
    row = {column: 0 for column in columns}

    row['credit.policy'] = int(user_input['credit_policy'])
    row['int.rate'] = user_input['int_rate'] / 100.0
    row['installment'] = user_input['installment']
    row['log.annual.inc'] = user_input['log_annual_inc']
    row['dti'] = user_input['dti']
    row['fico'] = user_input['fico']
    row['days.with.cr.line'] = user_input['days_with_cr_line']
    row['revol.bal'] = user_input['revol_bal']
    row['revol.util'] = user_input['revol_util']
    row['inq.last.6mths'] = user_input['inq_last_6mths']
    row['delinq.2yrs'] = user_input['delinq_2yrs']
    row['pub.rec'] = user_input['pub_rec']

    purpose_col = f"purpose_{user_input['purpose']}"
    if purpose_col in row:
        row[purpose_col] = 1

    return pd.DataFrame([row], columns=columns)


st.title('Loan Risk Prediction App')
st.caption('Predict whether a borrower is likely to fully repay their loan or default.')

with st.sidebar:
    st.header('Enter loan details')

    user_input = {
        'credit_policy': st.selectbox('Credit policy', [0, 1]),
        'purpose': st.selectbox(
            'Loan purpose',
            ['credit_card', 'debt_consolidation', 'educational', 'major_purchase', 'small_business', 'all_other'],
            index=1,
        ),
        'int_rate': st.slider('Interest rate (%)', min_value=5.0, max_value=25.0, value=12.0, step=0.1),
        'installment': st.slider('Installment amount', min_value=50.0, max_value=1200.0, value=300.0, step=10.0),
        'log_annual_inc': st.slider('Log annual income', min_value=7.0, max_value=14.0, value=10.5, step=0.1),
        'dti': st.slider('Debt-to-income ratio', min_value=0.0, max_value=40.0, value=12.0, step=0.1),
        'fico': st.slider('FICO score', min_value=300, max_value=850, value=700, step=1),
        'days_with_cr_line': st.slider('Days with credit line', min_value=0.0, max_value=10000.0, value=1000.0, step=50.0),
        'revol_bal': st.slider('Revolving balance', min_value=0, max_value=200000, value=15000, step=100),
        'revol_util': st.slider('Revolving utilization (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0),
        'inq_last_6mths': st.slider('Inquiries last 6 months', min_value=0, max_value=10, value=1),
        'delinq_2yrs': st.slider('Delinquencies in past 2 years', min_value=0, max_value=10, value=0),
        'pub_rec': st.slider('Public records', min_value=0, max_value=10, value=0),
    }

    predict_button = st.button('Predict loan outcome')


model, X_train, X_test, y_train, y_test, preds = train_model()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Model type', 'Random Forest')
with col2:
    st.metric('Train rows', len(X_train))
with col3:
    st.metric('Test rows', len(X_test))

st.subheader('Model performance')
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

col4, col5 = st.columns(2)
with col4:
    st.metric('Accuracy', round(float(acc), 4))
with col5:
    st.dataframe(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1']))

st.text(classification_report(y_test, preds))

st.subheader('Prediction')
if predict_button:
    input_df = build_single_record(user_input)
    pred_value = int(model.predict(input_df)[0])
    prob = model.predict_proba(input_df)[0]
    risk_prob = float(prob[1])
    safe_prob = float(prob[0])

    if pred_value == 1:
        label = 'Likely not fully paid'
        color = '🔴'
    else:
        label = 'Likely fully paid'
        color = '🟢'

    st.markdown(f'## {color} {label}')
    st.progress(risk_prob)
    st.write(f'Probability of not fully paid: {risk_prob:.2%}')
    st.write(f'Probability of fully paid: {safe_prob:.2%}')
else:
    st.info('Click Predict loan outcome to see the result.')
