import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X,y=make_moons()
X_train,X_test,y_train,y_test=train_test_split(X,y)


st.sidebar.markdown("# Streamlit Dashboard")

dataset=st.sidebar.multiselect(
    'Dataset',
    ('DS1','DS2')
)

algorithm=st.sidebar.multiselect(
    'Algorithm',
    ('Gradient Boosting for Classification','XgBoost for Classification')
)

default=st.sidebar.checkbox('Default Hyper Parameter Values')

loss=st.sidebar.multiselect(
    'Loss Function',
    ('deviance','exponential')
)

learning_rate=st.sidebar.number_input('Learning Rate')

n_estimators=st.sidebar.slider('Boosting Stages')

subsample=st.sidebar.number_input('Subsample')

criterion=st.sidebar.multiselect(
    'Criterion',
    ('friedman_mse','mse','mae')
)

min_samples_split=st.sidebar.number_input('Minimum Samples Split')

min_samples_leaf=st.sidebar.number_input('Minimum Samples Leaf')

min_weight_fraction_leaf=st.sidebar.number_input('Minimum Weight Fraction')

max_depth=st.sidebar.slider('Maximum Depth')

min_impurity_decrease=st.sidebar.number_input('Minimum Impurity Decrease')

min_impurity_split=st.sidebar.number_input('Minimum Impurity Split')

init=st.sidebar.multiselect(
    'Init (Have to think)',
    ('estimator','zero')
)

random_state=st.sidebar.multiselect(
    'Random State (Have to think)',
    (' ',' ')
)

max_features=st.sidebar.multiselect(
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

# fig,ax=plt.subplots()
#
# ax.scatter(X.T[0],X.T[1],c=y,cmap='rainbow')
# st.pyplot(fig)

# st.pyplot(fig).empty()

if st.sidebar.button('Run Algorithm'):
    print('Hello! world')