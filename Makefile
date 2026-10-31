# Builds dwim and installs it and its man page under PREFIX:
#
#   make install                     # in ~/.local
#   make install PREFIX=/usr/local   # system-wide, which may need sudo
#
# `make` builds as you, so `make && sudo make install PREFIX=/usr/local`
# builds nothing as root.
#
# Each step prints one line such as `  INSTALL ~/.local/bin/dwim`.
# `make V=1` prints the commands instead.

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
MANDIR ?= $(PREFIX)/share/man

BINARY = target/release/dwim
SOURCES = Cargo.toml Cargo.lock rust-toolchain.toml \
	$(wildcard */Cargo.toml */*.rs */*/*.rs */*/*/*.rs gpu/*/kernels/*)

# $(Q) hides a command, and $(call msg,VERB,what) announces it. With V=1
# the command is shown instead; with `make -s` nothing is.
ifeq ($(V),1)
  Q =
  msg =
else
  Q = @
  msg = @printf '  %-7s %s\n' '$(1)' '$(2)'
endif
ifneq ($(findstring s,$(firstword -$(MAKEFLAGS))),)
  msg =
endif

all: $(BINARY)

$(BINARY): $(SOURCES)
	$(call msg,CARGO,$@)
	$(Q)cargo build --release --locked -p dwim-cli
	$(Q)touch $@

install: all
	$(call msg,INSTALL,$(DESTDIR)$(BINDIR)/dwim)
	$(Q)install -d $(DESTDIR)$(BINDIR)
	$(Q)install -m 755 $(BINARY) $(DESTDIR)$(BINDIR)/dwim
	$(call msg,INSTALL,$(DESTDIR)$(MANDIR)/man1/dwim.1)
	$(Q)install -d $(DESTDIR)$(MANDIR)/man1
	$(Q)install -m 644 man/dwim.1 $(DESTDIR)$(MANDIR)/man1/dwim.1

uninstall:
	$(call msg,RM,$(DESTDIR)$(BINDIR)/dwim)
	$(Q)rm -f $(DESTDIR)$(BINDIR)/dwim
	$(call msg,RM,$(DESTDIR)$(MANDIR)/man1/dwim.1)
	$(Q)rm -f $(DESTDIR)$(MANDIR)/man1/dwim.1

test:
	$(call msg,TEST,workspace)
	$(Q)cargo test

clean:
	$(call msg,CLEAN,target)
	$(Q)cargo clean

.PHONY: all install uninstall test clean
