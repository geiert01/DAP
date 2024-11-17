- bitte erklären:
$ nvidia-smi // to view GPU utilization 
$ htop // to view view CPU utilization 
$ htop -s PERCENT_MEM // to view view CPU utilization sorted my memory-usage 
$


https://wiki.univie.ac.at/display/DataMining/Server+Hardware+-+Getting+Started
https://wiki.univie.ac.at/display/DataMining/Server+Hardware+-+Best+Practice


# KALENDER
https://ucloud.univie.ac.at/index.php/apps/calendar/p/J8VPUW8OQTKMSZET/dayGridMonth/2024-11-01


jeden 10000 schritt im trinaing
mit neighborhood sampling berechnen


### DQN paper
file:///C:/Users/t_gei/Downloads/nature14236.pdf

#### TODO BIS 2 wochen
mnt/data ist local server
home directory ist gebackuped, begrenzter platz 
geiert01dm: 100GB platz

- (ein conda auf home, kann man auf allen server benutzen) -> 10 gb platz -> aufpassen was man da installiert und was nicht
  - manchmal .cache inhalt löschen (timo fragen wenn nötig, also kein speicher)
- mt/data: auch da conda rein installieren ABER NUR AUF EINEM SERVER. Aber kein stress mit speicher platz, mit mehr enviroments etc 



2 envs von interesse atari und deep mind control
bevor deepmindc bei timo melden

wir wollen env wo plasticity loss vorkommt:
- phoenix (atari) auf jedenfall
- demon attack (atari)
- asterix (atari)

deepmindcontrol:
- agent: doctorqv2 zum laufen zu bringen
  - was bedeutet für lipkon ob man mit oder ohne image augmentation trainiert?

für atari vorher swig installieren


- https://arxiv.org/abs/2105.05246 methode um lipschitz in local neighborhood auszurechnen

# code todo
dqn_atari von cleanrl dort probieren lipschitz zu tracken
lipschitz constante tracken für bilder: gibs was, gibs nix

- downsampling, normalisieren für bilder




# TImo frage
.folder wichtig in home?
noframeskip oder ohne noframeskip model? 
- ALE/Phoenix-v5
- PhoenixNoFrameskip-v4