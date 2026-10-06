"""Embed the latest data into the app and write site/ for GitHub Pages."""
import os, shutil
root = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(root, 'app.html')).read()
a = t.index('let DATA=') + len('let DATA=')
b = t.index(';\nlet P=DATA.players')
t = t[:a] + open(os.path.join(root, 'dist', 'data.json')).read() + t[b:]
head = ('<meta name="apple-mobile-web-app-capable" content="yes">\n'
        '<meta name="apple-mobile-web-app-title" content="Start/Sit">\n'
        '<meta name="theme-color" content="#2E6B45">\n'
        '<link rel="apple-touch-icon" href="icon.png">\n<link rel="icon" href="icon.png">\n')
if 'apple-touch-icon' not in t:
    t = t.replace('<title>', head + '<title>', 1)
os.makedirs(os.path.join(root, 'site'), exist_ok=True)
open(os.path.join(root, 'site', 'index.html'), 'w').write(t)
shutil.copy(os.path.join(root, 'icon.png'), os.path.join(root, 'site', 'icon.png'))
open(os.path.join(root, 'site', '.nojekyll'), 'w').write('')
print('site built', len(t) // 1024, 'KB')
