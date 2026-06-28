from topo_nca import UnifiedTopologyNCA

def main():
    # Run a quick test verification of the unified module
    print("Initializing UnifiedTopologyNCA(6, 6)...")
    nca = UnifiedTopologyNCA(6, 6)

    print("Assembling Sheaf Laplacian...")
    L_sheaf = nca.assemble_sheaf_laplacian()
    print(f"Sheaf Laplacian shape: {L_sheaf.shape}")

    print("Training shape (target_chi=0, target_mass=12, steps=1000)...")
    binary_grid, chi, mass = nca.train_shape(target_chi=0, target_mass=12, steps=1000)

    print("\nBinary Grid Output:")
    print(binary_grid)
    print(f"\nVerified Chi: {chi}")
    print(f"Verified Mass: {mass}")

if __name__ == "__main__":
    main()
