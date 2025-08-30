# TradeSense.spec

# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis(
    ['gui/gui_main.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('models/forex_model.pkl', 'models'),
        ('logs/trade_journal.csv', 'logs'),
        ('strategies/custom_logic.py', 'strategies'),
        ('temp/chart_ma.png', 'temp'),
        ('temp/chart_macd.png', 'temp'),
        ('temp/chart_pattern.png', 'temp'),
        
        
    ],
    hiddenimports=collect_submodules('core'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TradeSense',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False  # 👈 no console window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TradeSense'
)
