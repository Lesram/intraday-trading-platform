# 📂 VERSION CONTROL GUIDE
# How to manage your Trading Platform versions

## 🎯 CURRENT STABLE VERSION
Tag: v1.0-stable
Commit: cb2ceea
Features: Market timing, Real trading, Professional dashboard

## 🔄 BASIC GIT COMMANDS

### Save current work (create new version):
git add .
git commit -m "Description of changes"
git tag -a "v1.1-feature-name" -m "Description"

### Go back to stable version:
git checkout v1.0-stable

### Return to latest work:
git checkout master

### See all versions:
git tag
git log --oneline

### Create a new branch for experiments:
git checkout -b experimental-features
# Make changes...
git add .
git commit -m "Experimental changes"

### Switch back to stable:
git checkout master

### See what changed:
git diff v1.0-stable

## 🚀 QUICK RESTORE COMMANDS

# Emergency restore to stable version:
git checkout v1.0-stable -- .
git commit -m "Restored to stable version"

# Create backup before major changes:
git checkout -b backup-before-changes
git checkout master

## 📋 VERSION HISTORY
v1.0-stable: Working platform with market timing (August 2025)
