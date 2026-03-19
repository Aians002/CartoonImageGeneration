# Git & GitHub Standard Practices
*Company Engineering Guidelines for Version Control & Collaboration*

---

## 1. Purpose
This document defines the standard practices for using Git and GitHub within the organization. It ensures consistent code management, clean commit history, structured collaboration, and safe deployment practices across engineering teams.

## 2. Branching Strategy

### Main Branches
| Branch | Purpose |
| :--- | :--- |
| `main` | Production ready code |
| `develop` | Integration branch for features |
| `staging` | Pre-production testing |

### Supporting Branch Types
| Branch Type | Example | Purpose |
| :--- | :--- | :--- |
| `feature` | `feature/login-api` | New feature development |
| `bugfix` | `bugfix/login-crash` | Fix bugs |
| `hotfix` | `hotfix/payment-fix` | Urgent production fix |
| `release` | `release/v1.3.0` | Prepare release |

## 3. Branch Naming Convention
**Format:** `type/short-description`

- `feature/user-login`
- `feature/payment-integration`
- `bugfix/profile-image-upload`
- `hotfix/api-timeout`

## 4. Commit Message Convention
**Format:** `type(scope): short description`

| Commit Type | Purpose |
| :--- | :--- |
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructuring |
| `chore` | Maintenance |
| `docs` | Documentation |
| `test` | Unit tests |
| `perf` | Performance improvement |

## 5. Pull Request Process
1. Create feature branch from `develop`
2. Complete development
3. Push branch to GitHub
4. Create Pull Request to `develop`
5. Request review from at least 1–2 reviewers
6. Fix review comments
7. Merge after approval

## 6. Pull Request Checklist
- [ ] Code builds successfully
- [ ] No debug logs
- [ ] No commented code
- [ ] Proper error handling
- [ ] Unit tests added if required
- [ ] Code formatted

## 7. Code Quality Tools

| Technology | Tool | Purpose |
| :--- | :--- | :--- |
| General | SonarQube | Code quality analysis |
| JS / TS | ESLint | Linting |
| JS / TS | Prettier | Code formatting |
| Flutter / Dart | Dart Analyzer | Static analysis |
| Flutter / Dart | dart format | Code formatting |
| Python | Flake8 | Linting |
| Python | Black | Code formatting |
| Python | Pylint | Code quality analysis |
| Python | isort | Import sorting |

## 8. Security Best Practices
- Never commit API keys
- Never commit private keys
- Never commit passwords
- Never commit production configuration

## 9. Common Mistakes to Avoid
- Direct commits to `main`
- Very large pull requests
- Poor commit messages
- Ignoring review comments

## 10. Ownership & Responsibility
- Write clean commits
- Follow branching strategy
- Participate in code reviews
- Maintain code quality

## 11. Repository Naming Convention
To maintain consistency across all projects, repositories should follow naming conventions based on the technology ecosystem standards.

| Technology | Naming Style | Example |
| :--- | :--- | :--- |
| Node / React / Next | `kebab-case` | `cilans-auth-api` |
| Flutter / Dart | `snake_case` | `cilans_mobile_app` |
| Python | `snake_case` | `cilans_auth_service` |

## 12. Common Git Commands (Daily Development)
These commands are commonly used by developers during daily development work.

| Command | Description |
| :--- | :--- |
| `git clone <repo-url>` | Clone a repository from GitHub to local machine |
| `git status` | Show current changes and branch status |
| `git branch` | List all local branches |
| `git branch <branch-name>` | Create a new branch |
| `git checkout <branch-name>` | Switch to a branch |
| `git checkout -b <branch-name>` | Create and switch to a new branch |
| `git pull origin develop` | Get latest code from remote branch |
| `git add .` | Stage all changed files |
| `git add <file-name>` | Stage a specific file |
| `git commit -m "message"` | Commit staged changes with message |
| `git push origin <branch-name>` | Push branch to GitHub |
| `git fetch` | Download latest changes from remote without merging |
| `git merge <branch-name>` | Merge another branch into current branch |
| `git stash` | Temporarily save uncommitted changes |
| `git stash pop` | Restore stashed changes |
| `git log` | View commit history |
| `git diff` | Show file changes before commit |



