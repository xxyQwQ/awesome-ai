# latent_dims = [1, 2, 64]
python main.py checkpoint=checkpoint/latent_1 parameter.latent_dims=1
python main.py checkpoint=checkpoint/latent_2 parameter.latent_dims=2
python main.py checkpoint=checkpoint/latent_64 parameter.latent_dims=64
python plot.py # visualization

# beta_value = [0.1, 0.5, 1, 2, 10]
python main.py checkpoint=checkpoint/beta_0.1 parameter.beta_value=0.1
python main.py checkpoint=checkpoint/beta_0.5 parameter.beta_value=0.5
python main.py checkpoint=checkpoint/beta_1 parameter.beta_value=1
python main.py checkpoint=checkpoint/beta_2 parameter.beta_value=2
python main.py checkpoint=checkpoint/beta_10 parameter.beta_value=10
