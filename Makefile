# Builds dwim and installs it and its man page under PREFIX:
#
#   make install                     # in ~/.local
#   make install PREFIX=/usr/local   # system-wide, which may need sudo
#
# `make` builds as you, so `make && sudo make install PREFIX=/usr/local`
# builds nothing as root.

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
MANDIR ?= $(PREFIX)/share/man

BINARY = target/release/dwim
SOURCES = Cargo.toml Cargo.lock rust-toolchain.toml \
	$(wildcard */Cargo.toml */*.rs */*/*.rs */*/*/*.rs gpu/*/kernels/*)

all: $(BINARY)

$(BINARY): $(SOURCES)
	cargo build --release --locked -p dwim-cli
	@touch $(BINARY)

install: all
	install -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(MANDIR)/man1
	install -m 755 $(BINARY) $(DESTDIR)$(BINDIR)/dwim
	install -m 644 man/dwim.1 $(DESTDIR)$(MANDIR)/man1/dwim.1

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/dwim $(DESTDIR)$(MANDIR)/man1/dwim.1

test:
	cargo test

clean:
	cargo clean

.PHONY: all install uninstall test clean
