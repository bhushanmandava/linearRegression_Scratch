
# Linear Regression using Gradient Descent  

This Python script implements a simple Linear Regression model using Gradient Descent from scratch to predict salary based on experience.  

---

## Dataset  
The model uses the `Salary_Data.csv` dataset, which should be in the same directory as the script. The dataset should have two columns:
- **Experience**: Years of experience (independent variable)
- **Salary**: Corresponding salary (dependent variable)  

---

## Prerequisites  

Make sure you have the following libraries installed:  
```bash
pip install pandas
```

---

## How It Works  

1. **Gradient Descent Algorithm**:  
   The `gradient_descent()` function updates the model's parameters (`m` and `b`) by calculating the gradients using the following equations:
   - \( m = m - \text{learning_rate} \times m\_gradient \)
   - \( b = b - \text{learning_rate} \times b\_gradient \)

2. **Loss Function**:  
   The `loss()` function calculates the Mean Squared Error (MSE) to evaluate the model's performance.

3. **Training the Model**:  
   The model is trained for a specified number of epochs (`epoch=300`), and the loss is printed every 50 epochs.

---

## Script Explanation  

```python
def gradient_descent(m1, b1, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        m_gradient += -(2/n) * (y - (m1 * x + b1)) * x
        b_gradient += -(2/n) * (y - (m1 * x + b1))
    m = m1 - learning_rate * m_gradient
    b = b1 - learning_rate * b_gradient
    return m, b
```

- **`gradient_descent()`**: Updates the model parameters `m` (slope) and `b` (intercept) using the gradients calculated over the dataset.

```python
def loss(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))
```

- **`loss()`**: Calculates Mean Squared Error (MSE) to measure the model's performance.

---

## How to Run  

1. Ensure `Salary_Data.csv` is in the same directory.  
2. Run the script using:  
    ```bash
    python linear_regression.py
    ```

---

## Output  

- The script prints the loss every 50 epochs to show the training progress.  
- The final values of `m` (slope) and `b` (intercept) are displayed at the end.

---

## Example Output  

```
epoch: 0  
loss: 23456.789  
epoch: 50  
loss: 1234.567  
epoch: 100  
loss: 789.123  
...  
Final parameters: m = 1.234, b = 567.890  
```

