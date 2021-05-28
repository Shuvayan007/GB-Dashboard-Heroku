import streamlit
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

X,y=make_moons(n_samples=300,noise=0.2,random_state=20)
X_train,X_test,y_train,y_test=train_test_split(X,y)


st.sidebar.markdown("# Streamlit Dashboard")

dataset=st.sidebar.selectbox(
    'Dataset',
    ('DS1','DS2')
)

algorithm=st.sidebar.selectbox(
    'Algorithm',
    ('Gradient Boosting for Classification','XgBoost for Classification')
)

default=st.sidebar.checkbox('Default Hyper Parameter Values')

loss=st.sidebar.selectbox(
    'Loss Function',
    ('deviance','exponential')
)

learning_rate=st.sidebar.number_input('Learning Rate')

n_estimators=st.sidebar.slider('Boosting Stages')

subsample=st.sidebar.number_input('Subsample')

criterion=st.sidebar.selectbox(
    'Criterion',
    ('friedman_mse','mse','mae')
)

min_samples_split=st.sidebar.number_input('Minimum Samples Split')

min_samples_leaf=st.sidebar.number_input('Minimum Samples Leaf')

min_weight_fraction_leaf=st.sidebar.number_input('Minimum Weight Fraction')

max_depth=st.sidebar.slider('Maximum Depth')

min_impurity_decrease=st.sidebar.number_input('Minimum Impurity Decrease')

min_impurity_split=st.sidebar.number_input('Minimum Impurity Split')

init=st.sidebar.selectbox(
    'Init (Have to think)',
    ('estimator','zero')
)

random_state=st.sidebar.selectbox(
    'Random State (Have to think)',
    (' ',' ')
)

max_features=st.sidebar.selectbox(
    'Max Features',
    ('auto','sqrt','log2')
)

verbose=st.sidebar.slider('Verbose')

max_leaf_nodes=st.sidebar.slider('Maximum Leaf Nodes')

warm_start=st.sidebar.multiselect(
    'Warm Start',
    ('True','False')
)

validation_fraction=st.sidebar.number_input('Validation Fraction')

n_iter_no_change= st.sidebar.slider('n Iteration No Change')

tol=st.sidebar.number_input('Tolerance')

ccp_alpha=st.sidebar.number_input('Cost-Complexity Pruning Alpha')

fig,ax=plt.subplots()

ax.scatter(X.T[0],X.T[1],c=y,cmap='rainbow')
orig=st.pyplot(fig)


if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf=GradientBoostingClassifier()
    clf.fit(X_train,y_train)
    streamlit.spinner('Your model is getting trained..')
    y_pred=clf.predict(X_test)

    XX,YY,input_array=draw_meshgrid()
    labels=clf.predict(input_array)

    ax.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.3,cmap='rainbow')

    plt.xlabel('Col1')
    plt.ylabel('Col2')
    orig=st.pyplot(fig)
    st.sidebar.subheader("Accuracy of the model: "+str(round(accuracy_score(y_test,y_pred),2)))