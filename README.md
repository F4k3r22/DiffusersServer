<h1 align="center">DiffusersServer</h1>

<div align="center">
  <img src="static/Diffusers_Server.png" alt="DiffusersServer Logo" width="200"/>
</div>

<p align="center">
  🚀 Nueva solución tipo Ollama, pero diseñada específicamente para modelos de generación de imágenes (Text-to-Image).
</p>

---

## 🌟 ¿Qué es DiffusersServer?

**DiffusersServer** es un servidor de inferencia basado en Flask y Waitress que permite generar imágenes a partir de texto (*Text-to-Image*) utilizando modelos avanzados de difusión.

Compatible con **Stable Diffusion 3**, **Stable Diffusion 3.5**, **Flux**, y **Stable Diffusion v1.5**, proporciona una API REST eficiente para integrar generación de imágenes en tus aplicaciones.

## ⚡ Características principales

✅ **Soporte para múltiples modelos**

- Stable Diffusion 3 *(Medium)*
- Stable Diffusion 3.5 *(Large, Large-Turbo, Medium)*
- Flux *(Flux 1 Schnell, Flux 1 Dev)*
- Stable Diffusion v1.5

✅ **Compatibilidad con GPU y MPS**

- Aceleración con CUDA (GPUs NVIDIA)
- Compatibilidad con MPS (Macs con chips M1/M2)

✅ **Servidor eficiente y escalable**

- Implementación con Flask + Waitress
- Soporte para múltiples hilos
- Carga los modelos en memoria una sola vez

✅ **API REST fácil de usar**

- Endpoint para inferencia: `POST /api/inference`
- Parámetros personalizables: prompt, modelo, tamaño de imagen, cantidad de imágenes

✅ **Gestión optimizada de memoria**

- *CPU offloading* en modelos Flux para reducir uso de VRAM
- Monitoreo opcional de consumo de memoria

---

## 🚀 DiffusersServer está diseñado para ofrecer una solución ligera, rápida y flexible para la generación de imágenes a partir de texto.

Si te gusta el proyecto, ¡considera darle una ⭐!

---

## 🚀 Planes a Futuro

Estamos trabajando en la integración de una API para modelos Text-to-Video (T2V), comenzando con Wan 2.1. Esto permitirá generar videos a partir de texto, ampliando las capacidades de DiffusersServer más allá de la generación de imágenes.

Tambien estamos trabajando en una mejor integración en los modelos pre existente T2Img de Diffusers

---

# Donaciones 💸

Si deseas apoyar este proyecto, puedes hacer una donación a través de PayPal:

[![Donate with PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate?hosted_button_id=KZZ88H2ME98ZG)

Tu donativo permite mantener y expandir nuestros proyectos de código abierto en beneficio de toda la comunidad.
