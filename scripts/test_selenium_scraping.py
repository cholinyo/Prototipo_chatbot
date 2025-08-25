#!/usr/bin/env python3
"""
Script para diagnosticar y solucionar problemas de Selenium en Windows
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import sys
import subprocess
import platform
import requests
import zipfile
from pathlib import Path
import json

def print_system_info():
    """Imprimir información del sistema"""
    print("🖥️ INFORMACIÓN DEL SISTEMA")
    print("=" * 40)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Arquitectura: {platform.architecture()[0]}")
    print(f"Python: {sys.version}")
    print(f"Python ejecutable: {sys.executable}")
    print()

def check_chrome_installation():
    """Verificar instalación de Chrome"""
    print("🔍 VERIFICANDO CHROME...")
    
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
    ]
    
    chrome_found = False
    chrome_path = None
    
    for path in chrome_paths:
        if os.path.exists(path):
            chrome_found = True
            chrome_path = path
            print(f"✅ Chrome encontrado en: {path}")
            
            # Obtener versión de Chrome
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    print(f"   Versión: {version}")
            except:
                print("   No se pudo obtener la versión")
            break
    
    if not chrome_found:
        print("❌ Chrome no encontrado")
        print("💡 Instala Google Chrome desde: https://www.google.com/chrome/")
        return False, None
    
    return True, chrome_path

def clean_webdriver_cache():
    """Limpiar cache de webdriver-manager"""
    print("\n🧹 LIMPIANDO CACHE DE WEBDRIVER...")
    
    cache_dirs = [
        os.path.expanduser("~/.wdm"),
        os.path.expanduser("~/AppData/Local/.wdm"),
        os.path.expanduser("~/AppData/Roaming/.wdm")
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print(f"✅ Cache eliminado: {cache_dir}")
            except Exception as e:
                print(f"⚠️ No se pudo eliminar {cache_dir}: {e}")

def download_chromedriver_manually():
    """Descargar ChromeDriver manualmente"""
    print("\n📥 DESCARGANDO CHROMEDRIVER MANUALMENTE...")
    
    try:
        # Detectar versión de Chrome
        chrome_version = get_chrome_version()
        if not chrome_version:
            print("❌ No se pudo detectar la versión de Chrome")
            return None
        
        print(f"   Versión de Chrome detectada: {chrome_version}")
        
        # Determinar versión de ChromeDriver compatible
        major_version = chrome_version.split('.')[0]
        
        # URLs para diferentes versiones
        if int(major_version) >= 115:
            # Usar Chrome for Testing API para versiones nuevas
            api_url = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
            response = requests.get(api_url, timeout=30)
            data = response.json()
            
            # Buscar versión compatible
            compatible_version = None
            for version_info in reversed(data['versions']):
                if version_info['version'].startswith(major_version):
                    downloads = version_info.get('downloads', {})
                    if 'chromedriver' in downloads:
                        for download in downloads['chromedriver']:
                            if download['platform'] == 'win32':
                                compatible_version = version_info['version']
                                driver_url = download['url']
                                break
                    if compatible_version:
                        break
        else:
            # Usar método legacy para versiones antiguas
            driver_url = f"https://chromedriver.storage.googleapis.com/LATEST_RELEASE_{major_version}"
            response = requests.get(driver_url, timeout=10)
            compatible_version = response.text.strip()
            driver_url = f"https://chromedriver.storage.googleapis.com/{compatible_version}/chromedriver_win32.zip"
        
        if not compatible_version:
            print("❌ No se encontró versión compatible de ChromeDriver")
            return None
        
        print(f"   Descargando ChromeDriver {compatible_version}...")
        
        # Crear directorio para driver
        driver_dir = Path("drivers")
        driver_dir.mkdir(exist_ok=True)
        
        # Descargar
        response = requests.get(driver_url, timeout=60)
        zip_path = driver_dir / "chromedriver.zip"
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extraer
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(driver_dir)
        
        # Buscar el ejecutable
        driver_exe = None
        for file_path in driver_dir.rglob("chromedriver.exe"):
            driver_exe = file_path
            break
        
        if driver_exe and driver_exe.exists():
            print(f"✅ ChromeDriver descargado: {driver_exe}")
            return str(driver_exe)
        else:
            print("❌ No se encontró chromedriver.exe después de la descarga")
            return None
            
    except Exception as e:
        print(f"❌ Error descargando ChromeDriver: {e}")
        return None

def get_chrome_version():
    """Obtener versión de Chrome"""
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
    ]
    
    for chrome_path in chrome_paths:
        if os.path.exists(chrome_path):
            try:
                result = subprocess.run([chrome_path, "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    version_text = result.stdout.strip()
                    # Extraer número de versión
                    import re
                    match = re.search(r'(\d+\.\d+\.\d+\.\d+)', version_text)
                    if match:
                        return match.group(1)
            except:
                continue
    return None

def test_selenium_with_custom_driver(driver_path):
    """Probar Selenium con driver personalizado"""
    print(f"\n🧪 PROBANDO SELENIUM CON DRIVER PERSONALIZADO...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        
        # Configurar opciones
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # Crear servicio con driver personalizado
        service = Service(driver_path)
        
        # Crear driver
        driver = webdriver.Chrome(service=service, options=options)
        
        # Test
        driver.get("https://httpbin.org/html")
        title = driver.title
        driver.quit()
        
        print(f"✅ Test exitoso - Título: {title}")
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False

def test_selenium_with_webdriver_manager():
    """Probar Selenium con webdriver-manager"""
    print("\n🧪 PROBANDO SELENIUM CON WEBDRIVER-MANAGER...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        # Configurar opciones
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        # Forzar descarga nueva
        driver_path = ChromeDriverManager().install()
        print(f"   Driver instalado en: {driver_path}")
        
        # Crear driver
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        
        # Test
        driver.get("https://httpbin.org/html")
        title = driver.title
        driver.quit()
        
        print(f"✅ Test exitoso - Título: {title}")
        return True, driver_path
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False, None

def create_selenium_config():
    """Crear configuración de Selenium para el proyecto"""
    config = {
        "selenium_config": {
            "webdriver_manager_enabled": False,
            "custom_driver_path": None,
            "chrome_options": [
                "--headless",
                "--no-sandbox", 
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1920,1080",
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ],
            "page_load_timeout": 30,
            "implicit_wait": 10,
            "explicit_wait": 15
        }
    }
    
    config_file = Path("config/selenium_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuración guardada en: {config_file}")
    return config_file

def try_firefox_alternative():
    """Probar Firefox como alternativa"""
    print("\n🦊 PROBANDO FIREFOX COMO ALTERNATIVA...")
    
    try:
        # Instalar geckodriver
        subprocess.run([sys.executable, "-m", "pip", "install", "webdriver-manager[firefox]"], 
                      check=True, capture_output=True)
        
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from webdriver_manager.firefox import GeckoDriverManager
        from selenium.webdriver.firefox.service import Service
        
        # Configurar opciones
        options = Options()
        options.add_argument('--headless')
        
        # Crear driver
        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=options)
        
        # Test
        driver.get("https://httpbin.org/html")
        title = driver.title
        driver.quit()
        
        print(f"✅ Firefox funcionando - Título: {title}")
        return True
        
    except Exception as e:
        print(f"❌ Firefox no funciona: {e}")
        return False

def main():
    """Función principal"""
    print_system_info()
    
    # Step 1: Verificar Chrome
    chrome_ok, chrome_path = check_chrome_installation()
    if not chrome_ok:
        print("\n❌ SOLUCIÓN: Instala Google Chrome primero")
        return
    
    # Step 2: Limpiar cache
    clean_webdriver_cache()
    
    # Step 3: Probar webdriver-manager
    print("\n" + "="*50)
    print("MÉTODO 1: WEBDRIVER-MANAGER")
    print("="*50)
    
    wdm_success, wdm_driver_path = test_selenium_with_webdriver_manager()
    
    if wdm_success:
        print("\n🎉 ¡ÉXITO! Selenium funcionando con webdriver-manager")
        config_file = create_selenium_config()
        
        # Actualizar config con ruta del driver
        with open(config_file, 'r') as f:
            config = json.load(f)
        config["selenium_config"]["webdriver_manager_enabled"] = True
        config["selenium_config"]["custom_driver_path"] = wdm_driver_path
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n✅ LISTO PARA USAR:")
        print("1. python scripts/test_selenium_scraping.py")
        print("2. python run_app.py")
        return
    
    # Step 4: Descargar manualmente
    print("\n" + "="*50)
    print("MÉTODO 2: DESCARGA MANUAL")
    print("="*50)
    
    manual_driver_path = download_chromedriver_manually()
    
    if manual_driver_path:
        if test_selenium_with_custom_driver(manual_driver_path):
            print("\n🎉 ¡ÉXITO! Selenium funcionando con driver manual")
            config_file = create_selenium_config()
            
            # Actualizar config
            with open(config_file, 'r') as f:
                config = json.load(f)
            config["selenium_config"]["webdriver_manager_enabled"] = False
            config["selenium_config"]["custom_driver_path"] = manual_driver_path
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n✅ CONFIGURACIÓN GUARDADA: {config_file}")
            print("\n✅ LISTO PARA USAR:")
            print("1. python scripts/test_selenium_scraping.py")
            print("2. python run_app.py")
            return
    
    # Step 5: Probar Firefox
    print("\n" + "="*50)
    print("MÉTODO 3: FIREFOX ALTERNATIVO")
    print("="*50)
    
    if try_firefox_alternative():
        print("\n🎉 ¡ÉXITO! Firefox funcionando como alternativa")
        print("\nNOTA: Necesitarás modificar el código para usar Firefox")
        return
    
    # Step 6: Modo sin Selenium
    print("\n" + "="*50)
    print("SOLUCIÓN: MODO SIN SELENIUM")
    print("="*50)
    print("❌ Selenium no funciona, pero puedes continuar:")
    print("1. El sistema usará solo requests para sitios compatibles")
    print("2. Documenta esta limitación en tu TFM") 
    print("3. Usa sitios estáticos para demostración")
    print("\n💡 SITIOS DE PRUEBA COMPATIBLES:")
    print("- https://httpbin.org/html")
    print("- https://example.com")
    print("- Sitios estáticos sin JavaScript")
    
    # Crear config sin Selenium
    config = {
        "selenium_config": {
            "webdriver_manager_enabled": False,
            "custom_driver_path": None,
            "selenium_available": False,
            "fallback_only_requests": True
        }
    }
    
    config_file = Path("config/selenium_config.json")
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuración sin Selenium guardada: {config_file}")

if __name__ == "__main__":
    main()