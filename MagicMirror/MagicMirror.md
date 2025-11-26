# MagicMirror² Smart Mirror

MagicMirror² is an open-source modular smart mirror platform that displays customizable information on a mirror-like screen. This project allows users to integrate real-time data such as time, weather, calendar events, and news feeds, while also supporting advanced features like facial landmark detection and AR overlays. In my setup, I have added custom modules including a rotating globe to visualize the world, skin analysis and AR makeup overlays for interactive beauty applications, and a configuration system that can be automated to start on boot using PM2. The modular architecture makes it easy to add, remove, or adjust modules according to personal preferences. This repository documents my installation process, configuration details, and custom module implementations, providing a complete guide for a personalized and interactive smart mirror experience.

## Table of Contents

| Section | Description |
|---------|-------------|
| Installation | Steps to download and install MagicMirror² and its dependencies |
| Configuration | Information on editing `config.js` to control modules and layout |
| Modules | List of default and installed modules, with brief descriptions |
| Custom Features | Details on custom implementations like Facial Landmark Detection, AR Makeup, Skin Analysis, and MMM-Globe |
| Autostart with PM2 | Instructions to use PM2 to automatically start MagicMirror² on boot and keep it running |
| Raspberry Pi OS Desktop Notes | Notes and tips for running MagicMirror² in the Raspberry Pi OS Desktop environment |
| Usage | How to run MagicMirror², customize modules, and add new features |
| References | Links to official documentation, module repositories, and related tutorials |

---

## Installation
1. Update the system
`sudo apt update`
`sudo apt upgrade -y`

2. Install dependencies
`curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -`

`sudo apt install -y nodejs git build-essential`

4. Clone MagicMirror²
`cd ~`

`git clone https://github.com/MichMich/MagicMirror`

`cd MagicMirror`

6. Install MagicMirror²
`npm install`

7.  Run MagicMirror²
`npm start`

## Configuration
- Main configuration file: `~/MagicMirror/config/config.js`
- Controls which modules are displayed, their positions, and individual settings
- Changes require restarting MagicMirror² to take effect

## Modules
- Clock – Displays current time
- Weather – Real-time weather information
- Calendar – Upcoming events
- Compliments – Rotating greeting messages
- MMM-Globe – Rotating world globe visualization

