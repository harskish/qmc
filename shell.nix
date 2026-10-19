let
  pkgs = import <nixpkgs> {
    config.allowUnfreePredicate = pkg: pkgs.lib.getName pkg == "pyviewer";
  };
  pyviewer = (import ../pyviewer/nix/python-packages.nix {
    inherit pkgs;
    pythonPackages = pkgs.python3Packages;
  }).pyviewer;
in
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [ ps.torch ps.numpy ps.tqdm pyviewer ]))
    pkgs.ninja
  ];

  shellHook = ''
    if [ -n "$WAYLAND_DISPLAY" ]; then
      export PYOPENGL_PLATFORM=egl
    fi
  '';
}
