# FRAGEN
- Email schon richtig gehandlet, wir haben nicht geantwortet aber wir haben passowrt bekommen

- vpn full tunnel, split tunnel etc? (big ip edge client)

- bitte erklären:
$ nvidia-smi // to view GPU utilization 
$ htop // to view view CPU utilization 
$ htop -s PERCENT_MEM // to view view CPU utilization sorted my memory-usage 
$

- conda installieren auf server
- welches conda?
- wie genau läuft das mit kalender?

https://wiki.univie.ac.at/display/DataMining/Server+Hardware+-+Getting+Started
https://wiki.univie.ac.at/display/DataMining/Server+Hardware+-+Best+Practice


# Methoden
## Lipschitz
- Spectral Normalization
- Weight Clipping mit matrix verändern vorm clippen
  - ist das was für uns?
- Almost orthogonal layers for efficient general ...



## Plasticity Loss
- Verschiedene Methoden zum Messen aus Paper
  file:///C:/Users/t_gei/Documents/Master/3.%20Semester/DAP/Zusammenhang%20zwischen%20Plasticity%20Loss%20und%20Data%20Augmentation.pdf
  - Weight norm
    - Weight norm widersprüchlich zu spectral norm?
    - Wenn wir spectral normalisieren, wird weight norm automatisch kleiner? Schlecht?
  - Feature rank
  - loss landscape
    -  Loss Landscape geben keinen exakten Wert zum Messen, wollen wir daher eher Methoden wie FAU?
  - Fraction of active units
    - Wurde auch im Paper verwendet, und gibt exakten Wert

- Schwierig informationen, Definitionen, Erklärungen zu finden
- Vor und Nachteile relevant?






jeden 10000 schritt im trinaing
mit neighborhood sampling berechnen

wie kann man lipschitz konstante aproxximieren?
versuchen zu implementieren (auf cartpole)
wie teuer ist es zu berechnen
noch literatur anschauen, um lipschitz konstante zu approximieren


terminal ls




### DQN paper
file:///C:/Users/t_gei/Downloads/nature14236.pdf

### Quantum paper
https://arxiv.org/pdf/2410.21117

cartpole lipschitz
https://arxiv.org/pdf/1705.08551
https://github.com/befelix/safe_learning/blob/master/examples/lyapunov_function_learning.ipynb
use 
https://arxiv.org/pdf/1312.6199 to calculate lipschitz


#####  Model based  RL - lipschitz continuity
https://arxiv.org/pdf/1804.07193