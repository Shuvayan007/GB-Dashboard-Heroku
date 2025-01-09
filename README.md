# Gradient Boosting Classifier Dashboard

Welcome to the **Gradient Boosting Classifier Dashboard**, a real-time interactive tool designed to help you visualize and experiment with the Gradient Boosting Classifier algorithm using the `make_moons` dummy dataset from scikit-learn. This project aims to provide an intuitive understanding of how different hyperparameters affect the performance of the algorithm.

---

## Features

- **Real-time Visualization:** Watch how the Gradient Boosting Classifier learns and adapts to the `make_moons` dataset.
- **Hyperparameter Tuning:** Modify hyperparameters such as:
  - Number of estimators
  - Learning rate
  - Maximum depth of trees
  - Subsample rate
  - Minimum samples split/leaf
- **Performance Metrics:** View the accuracy of the model as you tweak parameters.
- **User-Friendly Interface:** Built using Streamlit for an intuitive and interactive experience.

---

## Demo

Check out the live dashboard on Streamlit Community Cloud:

[**Gradient Boosting Classifier Dashboard**](<Insert-your-deployment-link-here>)

---

## Installation

To run the dashboard locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/gradient-boosting-dashboard.git
   cd gradient-boosting-dashboard
   ```

2. **Set up a virtual environment (optional):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Launch the Streamlit application either locally or using the provided demo link.
2. Adjust the hyperparameters using the sidebar controls.
3. Observe the changes in the decision boundary and accuracy in real-time.
4. Experiment with different combinations of hyperparameters to gain insights into their impact.

---

## Project Structure

```
.
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .streamlit           # Streamlit configuration files
```

---

## Dependencies

The following Python libraries are required:
- `scikit-learn`
- `numpy`
- `matplotlib`
- `pandas`
- `streamlit`

---

## About the Dataset

The `make_moons` dataset is a synthetic binary classification dataset available in scikit-learn. It consists of two interleaving half circles and is often used for visualization and demonstration purposes.

---

## Future Enhancements

- Include support for additional datasets.
- Add explanations and tooltips for hyperparameters.
- Provide feature importance visualization.

---

## Contributing

Contributions are welcome! If you have suggestions or want to add new features, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments

- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [Streamlit](https://streamlit.io/) for the easy-to-use app framework.

---

Happy experimenting with Gradient Boosting Classifier! :rocket: