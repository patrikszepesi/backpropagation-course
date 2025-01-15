#---------Forward Pass-----------
def forward_pass(x,w1,w2):
  h = x * w1
  y_hat = h * w2
  return h, y_hat


#------------Loss Function------------
def mse_loss(y,yhat):
  return 0.5 * (y-yhat)**2

#---------Gradients (Backprop)---------
def compute_gradients(x,w1,w2,y):
  h,yhat = forward_pass(x,w1,w2)
  dL_dy_hat = (yhat-y)
  dL_dw2 = dL_dy_hat * h
  dL_dw1 = dL_dy_hat * (x * w2)
  return dL_dw1, dL_dw2

#--------Training Loop-------

def train_network(x, y, w1_init, w2_init, alpha = 0.01, epochs = 10):

  w1, w2 = w1_init, w2_init

  for i in range(epochs):
    h, y_hat = forward_pass(x, w1, w2)
    loss_val = mse_loss(y, y_hat)

    dL_dw1, dL_dw2 = compute_gradients(x, w1, w2, y)

    print(f"Iteration {i+1:2d}: "
          f"w1={w1:7.3f}, w2={w2:7.3f}, "
          f"y_hat={y_hat:7.5f}, Loss={loss_val:8.5f}"
    )

    w1-= alpha * dL_dw1
    # w1_new = w1 - alpha * dL_dw1
    # w1 = w1_new
    w2-= alpha * dL_dw2
  return w1,w2


#----- Run------

if __name__ == "__main__":

  x = 2.0
  y = 20.0

  w1_init = 2.0
  w2_init = 0.5

  alpha = 0.01

  w1_final, w2_final = train_network(
      x, y, w1_init, w2_init,
      alpha = alpha,
      epochs = 10
  )

  print("Training Complete")
