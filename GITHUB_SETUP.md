# GitHub Repository Setup Guide

## 📦 Push to GitHub

### Step 1: Create GitHub Repository

1. Go to [github.com](https://github.com) and log in
2. Click the "+" icon in the top right → "New repository"
3. Repository name: `fiap-fase4-multiagent-video-analysis` (or your preferred name)
4. Description: `Multi-Agent Video Analysis System with CrewAI for facial recognition, emotion detection, and activity recognition`
5. **Keep it Public** (or Private if preferred)
6. **DO NOT** initialize with README (we already have one)
7. Click "Create repository"

### Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, run these commands:

```bash
# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### Example:

```bash
# Replace with your actual GitHub username and repository name
git remote add origin https://github.com/danilodecastro/fiap-fase4-multiagent-video-analysis.git
git branch -M main
git push -u origin main
```

### Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all files uploaded
3. The README.md will be displayed on the main page

## 🔒 Important: Before Pushing

Make sure sensitive files are NOT included (they're already in .gitignore):

✅ **Excluded** (in .gitignore):
- `.env` - Your API keys
- `.venv/` - Virtual environment
- `*.mp4` - Video files
- `output/` - Generated reports
- `__pycache__/` - Python cache

✅ **Included**:
- `.env.example` - Template without secrets
- All source code
- README and documentation
- Requirements.txt

## 🌿 Branch Strategy (Optional)

If you want to work with branches:

```bash
# Create development branch
git checkout -b develop

# Make changes and commit
git add .
git commit -m "Your changes"

# Push development branch
git push -u origin develop

# Create feature branches
git checkout -b feature/new-agent
# ... make changes ...
git add .
git commit -m "Add new agent feature"
git push -u origin feature/new-agent
```

## 🏷️ Tagging Releases (Optional)

```bash
# Create a tag for version 1.0
git tag -a v1.0 -m "Initial release: Multi-Agent Video Analysis System"

# Push tags to GitHub
git push origin v1.0

# Or push all tags
git push origin --tags
```

## 🔄 Future Updates

After making changes:

```bash
# Check status
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## 📝 Repository Structure on GitHub

Your repository will look like this:

```
fiap-fase4-multiagent-video-analysis/
├── README.md                    ← Will be displayed on GitHub homepage
├── QUICKSTART.md               ← Quick start guide
├── requirements.txt            ← Visible to all users
├── setup.sh                    ← Setup script
├── check_setup.py              ← Configuration checker
├── main.py                     ← Main application
├── agents/                     ← Agent implementations
├── tools/                      ← Custom tools
├── config/                     ← Configuration
├── aula1/                      ← Original facial recognition
├── images/                     ← Sample images
└── .gitignore                  ← Git ignore rules
```

## 🎯 GitHub Repository Settings

### Recommended Settings:

1. **About Section** (top right of repository):
   - Description: Multi-Agent Video Analysis System with CrewAI
   - Website: (your website if any)
   - Topics: `crewai`, `multi-agent`, `facial-recognition`, `emotion-detection`, `computer-vision`, `ai`, `python`, `mediapipe`, `deepface`, `groq`

2. **Branch Protection** (Settings → Branches):
   - Protect `main` branch
   - Require pull request reviews (if working in team)

3. **Issues**:
   - Enable Issues for bug tracking
   - Create issue templates (optional)

4. **Actions** (optional):
   - Can set up CI/CD for automated testing

## 📊 Add Badges to README (Optional)

Add these badges to the top of your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-0.70%2B-green)
![License](https://img.shields.io/badge/license-MIT-blue)
```

## 🤝 Collaboration

If working with a team:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME

# Install dependencies
./setup.sh

# Create feature branch
git checkout -b feature/your-feature

# Make changes, commit, and push
git add .
git commit -m "Your changes"
git push origin feature/your-feature

# Create Pull Request on GitHub
```

## 📱 GitHub Mobile

You can also manage your repository from the GitHub mobile app:
- Monitor commits
- Review code
- Merge pull requests
- Manage issues

## 🔗 Useful GitHub Commands

```bash
# View remote repositories
git remote -v

# Fetch latest changes
git fetch origin

# Pull latest changes
git pull origin main

# See commit history
git log --oneline

# See what changed
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

## 💡 Pro Tips

1. **Write good commit messages**: Be descriptive about what changed and why
2. **Commit often**: Small, focused commits are better than large ones
3. **Use branches**: Keep `main` stable, develop in feature branches
4. **Document changes**: Update README.md when adding new features
5. **Review before pushing**: Always check `git status` and `git diff` before committing

## ✅ Verification Checklist

Before pushing to GitHub:
- [ ] All sensitive files are in .gitignore
- [ ] README.md is complete and informative
- [ ] Requirements.txt is up to date
- [ ] Code is tested and working
- [ ] No hardcoded passwords or API keys
- [ ] .env.example is provided (without real keys)
- [ ] Documentation is clear

