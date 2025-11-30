# PokéAgent Challenge Documentation

This folder contains the project website for the PokéAgent Challenge submission.

## 🌐 Live Website

Once deployed via GitHub Pages, the website will be available at:
```
https://heatz123.github.io/pokeagent-solution/
```

## 📁 Structure

```
docs/
├── index.html              # Main webpage
├── static/
│   ├── css/
│   │   └── main.css       # Styling
│   ├── js/
│   │   └── main.js        # Interactive features
│   ├── images/
│   │   └── emerald.png    # Hero image
│   └── videos/
│       ├── final1_2x.mp4  # Demo video 1
│       └── final2_2x.mp4  # Demo video 2
└── README.md              # This file
```

## 🚀 Deployment (GitHub Pages)

### Step 1: Enable GitHub Pages

1. Go to repository **Settings**
2. Navigate to **Pages** (left sidebar)
3. Under **Source**, select:
   - Branch: `main`
   - Folder: `/docs`
4. Click **Save**

### Step 2: Wait for Deployment

GitHub will automatically build and deploy the site. This usually takes 1-2 minutes.

### Step 3: Access Your Site

Your site will be live at: `https://heatz123.github.io/pokeagent-solution/`

## 🎨 Customization

### Updating Content

- **Text content**: Edit `index.html`
- **Styling**: Modify `static/css/main.css`
- **Interactive features**: Update `static/js/main.js`

### Adding Videos

Place additional video files in `static/videos/` and reference them in `index.html`:

```html
<video controls preload="metadata">
    <source src="static/videos/your-video.mp4" type="video/mp4">
</video>
```

### Color Theme

The color scheme is defined in CSS variables (`:root` in `main.css`):

```css
--primary-color: #50C878;      /* Emerald Green */
--secondary-color: #2E8B57;    /* Dark Green */
--accent-color: #FFD700;       /* Gold */
```

## 📝 Features

- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Embedded video players with controls
- ✅ Smooth scroll animations
- ✅ One-click citation copy
- ✅ Scroll-to-top button
- ✅ SEO meta tags
- ✅ Open Graph tags for social sharing

## 🔧 Local Testing

To test the website locally:

1. **Option A: Simple HTTP Server (Python)**
   ```bash
   cd docs
   python3 -m http.server 8080
   ```
   Then open: http://localhost:8080

2. **Option B: Live Server (VS Code)**
   - Install "Live Server" extension
   - Right-click `index.html` → "Open with Live Server"

## 📊 Analytics (Optional)

To add Google Analytics, insert before `</head>` in `index.html`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## 🐛 Troubleshooting

**Videos not playing?**
- Ensure video files are in `static/videos/`
- Check browser console for errors
- Verify video codec (H.264 recommended)

**Page not updating after changes?**
- Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
- Wait a few minutes for GitHub Pages to rebuild

**CSS not loading?**
- Check file paths are relative: `static/css/main.css`
- Ensure files are committed to `main` branch

## 📄 License

MIT License - see main repository LICENSE file

---

**Author**: Junik Bae (배준익)
**Affiliation**: Seoul National University
**Year**: 2025
