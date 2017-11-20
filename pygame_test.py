import pygame
import time

filename = "C:\\Users\\NiWa\\Desktop\\music\\musicradar-drum-samples\\Drum Kits\\Kit 2 - Acoustic room\\CYCdh_K2room_Kick-06.wav"

pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
song = pygame.mixer.Sound(filename)
for i in range(10):
    pygame.mixer.Sound.play(song)
    #song.play()
    time.sleep(0.5)