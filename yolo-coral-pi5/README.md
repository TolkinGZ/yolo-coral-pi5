# YOLOv8 + Google Coral USB на Raspberry Pi 5

Проект для быстрого инференса моделей YOLOv8 на Raspberry Pi 5 с использованием аппаратного ускорителя Google Coral Edge TPU. Скрипт использует многопоточность для захвата видео, что позволяет достигать 30-75 FPS.

## Требования к оборудованию
- Raspberry Pi 5 (ОС: Raspberry Pi OS Bookworm 64-bit)
- Google Coral USB Accelerator
- USB Веб-камера

## Инструкция по установке на Raspberry Pi 5

### 1. Установка драйверов Google Coral
Подключите Coral к синему порту USB 3.0. Выполните в терминале:
```bash
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg
echo "deb[signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install libedgetpu1-std -y
