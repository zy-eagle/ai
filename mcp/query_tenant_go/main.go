package main

import (
	"fmt"
	"os"
)

func main() {
	server, err := CreateServer()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start MCP query server: %v\n", err)
		os.Exit(1)
	}

	if err := server.Serve(); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
