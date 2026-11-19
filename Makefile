# Builds dwim and installs it and its man page under PREFIX:
#
#   make install                     # in ~/.local
#   make install PREFIX=/usr/local   # system-wide, which may need sudo
#
# `make` builds as you, so `make && sudo make install PREFIX=/usr/local`
# builds nothing as root.
#
# `make check` checks the Lean specs in specs/, their definitions, their
# examples and their proofs; makes the pages that show them, in
# specs/.lake/build/literate-html/index.html; and checks that the code
# does what the specs say, with difftest.
#
# Each step prints one line such as `  INSTALL ~/.local/bin/dwim`.
# `make V=1` prints the commands instead.

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
MANDIR ?= $(PREFIX)/share/man

BINARY = target/release/dwim
SOURCES = Cargo.toml Cargo.lock rust-toolchain.toml build.rs lib.rs \
	$(wildcard */*.rs */*/*.rs */*/*/*.rs gpu/*/kernels/*)

# $(Q) hides a command, and $(call msg,VERB,what) announces it. With V=1
# the command is shown instead; with `make -s` nothing is.
ifeq ($(V),1)
  Q =
  msg =
else
  Q = @
  msg = @printf '  %-8s %s\n' '$(1)' '$(2)'
endif
# $(call quiet,command) shows what the command printed only if it fails.
quiet = out=$$($(1) 2>&1) || { printf '%s\n' "$$out"; exit 1; }

ifneq ($(findstring s,$(firstword -$(MAKEFLAGS))),)
  msg =
endif

all: $(BINARY)

$(BINARY): $(SOURCES)
	$(call msg,CARGO,$@)
	$(Q)cargo build --release --locked
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
	$(call msg,TEST,dwim)
	$(Q)cargo test

check: difftest-compaction
	$(call msg,LAKE,specs/.lake/build/literate-html/index.html)
	@# Verso does not notice changes to images and style.css: without its
	@# trace, it makes the pages again, which takes a second.
	$(Q)rm -f specs/.lake/build/literate-html.trace
	$(Q)cd specs && lake --quiet query :literateHtml > /dev/null

# `make difftest-SUBJECT` checks one subject against its reference, as
# `difftest SUBJECT` does: `make difftest-compaction`, `make difftest-model`.
difftest-%:
	$(call msg,DIFFTEST,$*)
	$(Q)$(call quiet,cargo run --quiet --bin difftest $*)

clean:
	$(call msg,CLEAN,target)
	$(Q)cargo clean

.PHONY: all install uninstall test check clean
