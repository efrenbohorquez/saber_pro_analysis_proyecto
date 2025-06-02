#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para convertir el documento de evidencia Markdown a PDF
"""

import markdown
from pathlib import Path
import webbrowser
import os

def markdown_to_html():
    """Convierte el archivo Markdown a HTML con estilos CSS."""
    
    # Leer el archivo markdown
    with open('EVIDENCIA_PROYECTO_CODIGO.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convertir markdown a HTML
    html = markdown.markdown(markdown_content, extensions=['codehilite', 'fenced_code', 'tables'])
    
    # CSS para mejor formato del PDF
    css = '''
    <style>
    @page {
        margin: 2cm;
        size: A4;
    }
    body {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        color: #333;
        margin: 0;
        padding: 20px;
        font-size: 12px;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        font-size: 24px;
        page-break-before: auto;
        margin-top: 30px;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        font-size: 20px;
        margin-top: 25px;
    }
    h3 {
        color: #34495e;
        font-size: 16px;
        margin-top: 20px;
        border-left: 4px solid #3498db;
        padding-left: 10px;
    }
    h4 {
        color: #2c3e50;
        font-size: 14px;
        margin-top: 15px;
    }
    code {
        background-color: #f8f9fa;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #e74c3c;
    }
    pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
        border-left: 4px solid #3498db;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        line-height: 1.4;
        page-break-inside: avoid;
    }
    pre code {
        background: none;
        padding: 0;
        color: #333;
    }
    blockquote {
        border-left: 4px solid #e74c3c;
        margin: 10px 0;
        padding-left: 15px;
        color: #666;
        font-style: italic;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        font-size: 11px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .page-break {
        page-break-before: always;
    }
    ul, ol {
        margin: 10px 0;
        padding-left: 20px;
    }
    li {
        margin: 5px 0;
    }
    p {
        margin: 10px 0;
        text-align: justify;
    }
    </style>
    '''
    
    # Crear HTML completo
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Evidencia del Proyecto - Análisis Saber Pro</title>
    {css}
</head>
<body>
    {html}
</body>
</html>'''
    
    # Guardar HTML
    output_file = 'EVIDENCIA_PROYECTO_CODIGO.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Archivo HTML creado: {output_file}")
    print(f"📂 Ubicación: {os.path.abspath(output_file)}")
    
    return output_file

def main():
    """Función principal."""
    print("🔄 Convirtiendo documento de evidencia a HTML...")
    
    try:
        html_file = markdown_to_html()
        
        print("\n📋 Instrucciones para generar PDF:")
        print("1. Abra el archivo HTML en su navegador")
        print("2. Use Ctrl+P (o Cmd+P en Mac)")
        print("3. Seleccione 'Guardar como PDF' o 'Microsoft Print to PDF'")
        print("4. Configure:")
        print("   - Orientación: Vertical")
        print("   - Tamaño: A4")
        print("   - Márgenes: Predeterminados")
        print("   - Más opciones: Incluir gráficos de fondo")
        
        # Abrir automáticamente en el navegador
        if input("\n¿Desea abrir el archivo HTML ahora? (s/n): ").lower().startswith('s'):
            webbrowser.open(os.path.abspath(html_file))
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
