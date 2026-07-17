import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# preparing real data 
def get_real_samples(n=100):
    x = torch.rand(n, 1) * 2 - 1
    y = x ** 2
    return torch.cat((x, y), dim=1)

#here will be discriminator 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 15),
            nn.ReLU(),
            nn.Linear(15, 2)
        )

    def forward(self, x):
        return self.model(x)


discriminator = Discriminator()
generator = Generator()

loss_function = nn.BCELoss()
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.001)

epochs = 5000
batch_size = 32

print("Начинаем обучение... Это займет пару секунд.")

for epoch in range(epochs):
   
    optimizer_d.zero_grad()
    
    real_samples = get_real_samples(batch_size)
    real_labels = torch.ones(batch_size, 1)
    output_d_real = discriminator(real_samples)
    loss_d_real = loss_function(output_d_real, real_labels)
    
    latent_space_samples = torch.randn(batch_size, 5)
    fake_samples = generator(latent_space_samples)
    fake_labels = torch.zeros(batch_size, 1)
    output_d_fake = discriminator(fake_samples.detach())
    loss_d_fake = loss_function(output_d_fake, fake_labels)
    
    loss_d = loss_d_real + loss_d_fake
    loss_d.backward()
    optimizer_d.step()
    
    # Учим Генератор
    optimizer_g.zero_grad()
    
    output_d_fake_for_g = discriminator(fake_samples)
    loss_g = loss_function(output_d_fake_for_g, real_labels) 
    
    loss_g.backward()
    optimizer_g.step()
    
    #
    if epoch % 1000 == 0:
        print(f"Эпоха {epoch}/{epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")


print("Обучение завершено! Генерируем точки и строим график...")

# Генерируем 100 фейковых точек
test_noise = torch.randn(100, 5)
generated_points = generator(test_noise).detach().numpy()

# Генерируем 100 настоящих точек для сравнения
real_points = get_real_samples(100).numpy()

# Строим график
plt.figure(figsize=(8, 6))
plt.scatter(real_points[:, 0], real_points[:, 1], color='blue', label='Настоящие данные ($y=x^2$)', alpha=0.5)
plt.scatter(generated_points[:, 0], generated_points[:, 1], color='red', label='Сгенерированные данные', alpha=0.5)
plt.legend()
plt.title("Результат работы Генератора")
plt.grid(True)
plt.show()