{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            bashInteractive
            gcc
            just
            meson
            ninja
          ];
        };
        packages.default = pkgs.stdenv.mkDerivation {
          name = "neural";
          src = ./.;
          nativeBuildInputs = with pkgs; [
            gcc
            meson
            ninja
          ];
          mesonFlags = [ "--buildtype" "release" ];
        };
        checks.default = pkgs.runCommand "check"
          { buildInputs = [ self.packages.${system}.default ]; }
          ''
            neural test
            echo ✅ all tests passed > $out
          '';
      }
    );
}
