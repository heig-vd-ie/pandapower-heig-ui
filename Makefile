# Common Makefile

PYTHON_VERSION := 3.12
VENV_DIR := .venv
ORG := heig-vd-ie

# Default target: help
.DEFAULT_GOAL := install-all


install-uv: ## Install uv (fast Python package manager)
	@echo "Checking if uv is installed..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv is already installed"; \
		uv --version; \
	fi

install-python-wsl: ## Install Python $(PYTHON_VERSION) and venv support on WSL
	@echo "Checking if Python $(PYTHON_VERSION) is installed..."
	@if ! command -v python$(PYTHON_VERSION) >/dev/null 2>&1; then \
		echo "Installing Python $(PYTHON_VERSION)..."; \
		echo "# Reference: Tutorial is the following link, https://www.linuxtuto.com/how-to-install-python-3-12-on-ubuntu-22-04/"; \
		sudo add-apt-repository -y ppa:deadsnakes/ppa; \
		sudo apt update; \
		sudo apt install -y python$(PYTHON_VERSION) python$(PYTHON_VERSION)-venv; \
	else \
		echo "Python $(PYTHON_VERSION) already installed"; \
	fi


install-deps: ## Install system dependencies
	@echo "Installing system dependencies..."
	@read -p "This will install system dependencies (libpq-dev gcc python3-dev build-essential direnv). Continue? [y/N] " answer; \
	if [ "$$answer" = "y" ] || [ "$$answer" = "Y" ]; then \
		sudo apt update; \
		sudo apt install -y libpq-dev gcc python3-dev build-essential direnv; \
	else \
		echo "Skipped installing dependencies."; \
	fi

_uv-venv: ## Create a virtual environment using uv
	@echo "Creating virtual environment using uv..."
	@command -v uv >/dev/null 2>&1 || (echo "uv is not installed. Run 'make install-uv' first."; exit 1)
	uv venv .venv --python $(PYTHON_VERSION)

venv-activate: SHELL:=/bin/bash
venv-activate: ## enter venv in a subshell
	@test -d .venv || make _venv
	@bash --rcfile <(echo '. ~/.bashrc; . .venv/bin/activate; echo "You are now in a subshell with venv activated."; . scripts/enable-direnv.sh') -i

uv-install: ## Install Python packages using uv
	@echo "Installing Python packages using uv..."
	@command -v uv >/dev/null 2>&1 || (echo "uv is not installed. Run 'make install-uv' first."; exit 1)
	uv pip install -e .

uv-sync: ## Sync dependencies using uv
	@echo "Syncing Python packages using uv..."
	@command -v uv >/dev/null 2>&1 || (echo "uv is not installed. Run 'make install-uv' first."; exit 1)
	uv sync --extra dev

uv-venv-setup: SHELL:=/bin/bash
uv-venv-setup: ## Setup venv and install packages using uv
	@echo "Setting up virtual environment and installing packages with uv..."
	@test -d .venv || make _uv-venv
	@bash --rcfile <(echo '. ~/.bashrc; . .venv/bin/activate; echo "You are now in a subshell with uv venv activated."; make uv-sync; make nbstripout-install; . scripts/enable-direnv.sh') -i

venv-activate-and-uv-install: SHELL:=/bin/bash
venv-activate-and-uv-install: ## Activate venv and install packages using uv (non-interactive)
	@echo "Activating virtual environment and installing packages with uv..."
	@test -d .venv || make _uv-venv
	@. .venv/bin/activate && make uv-sync && make nbstripout-install


install-all: ## Install all dependencies and set up the environment using uv
	@$(MAKE) install-uv
	@$(MAKE) install-python-wsl
	@$(MAKE) install-deps
	@$(MAKE) _uv-venv
	@$(MAKE) venv-activate-and-uv-install
	@echo "All dependencies installed successfully with uv!"

uninstall-venv: ## Uninstall the virtual environment
	@echo "Uninstalling virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment uninstalled."


nbstripout-install:
	@echo "Installing nbstripout git filter..."
	nbstripout --install