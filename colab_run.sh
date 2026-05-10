!wget "https://fem-on-colab.github.io/releases/fenicsx-install-release-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
!pip install optuna
!apt-get update
!apt-get install -qq xvfb libgl1-mesa-glx
!pip install pyvista -qq

!wget https://raw.githubusercontent.com/yapeansa/bioheat_femnn/refs/heads/main/mesh/malha.xdmf
!wget https://raw.githubusercontent.com/yapeansa/bioheat_femnn/refs/heads/main/mesh/malha.h5
!wget https://raw.githubusercontent.com/yapeansa/bioheat_femnn/refs/heads/main/mesh/malha_fina.xdmf
!wget https://raw.githubusercontent.com/yapeansa/bioheat_femnn/refs/heads/main/mesh/malha_fina.h5