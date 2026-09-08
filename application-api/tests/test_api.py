import pytest


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/live", {"status": "ok", "service": "application-api"}),
        ("/health", {"status": "ok"}),
    ],
)
def test_health_endpoints(client, path, expected):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == expected


def test_products_are_fixed(client):
    expected = [
        {"id": 1, "name": "Caderno", "price": 25.0, "available": True},
        {"id": 2, "name": "Caneta", "price": 5.0, "available": True},
        {"id": 3, "name": "Estojo", "price": 12.5, "available": True},
    ]
    for _ in range(2):
        response = client.get("/products")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json() == expected


@pytest.mark.parametrize(
    ("product_id", "quantity", "price", "total"),
    [(1, 2, 25.0, 50.0), (2, 1, 5.0, 5.0), (3, 3, 12.5, 37.5)],
)
def test_valid_order(client, product_id, quantity, price, total):
    products_before = client.get("/products").json()
    for _ in range(2):
        response = client.post(
            "/orders", json={"product_id": product_id, "quantity": quantity}
        )
        assert response.status_code == 201
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {
            "product_id": product_id,
            "quantity": quantity,
            "unit_price": price,
            "total": total,
            "status": "simulated",
        }
    assert client.get("/products").json() == products_before


@pytest.mark.parametrize("product_id", [0, -1, 999])
def test_unknown_product(client, product_id):
    response = client.post("/orders", json={"product_id": product_id, "quantity": 1})
    assert response.status_code == 404
    assert response.json() == {"detail": "Produto não encontrado"}


@pytest.mark.parametrize("quantity", [0, -1, 1.5, 2.0, "2", True, None])
def test_invalid_quantity(client, quantity):
    response = client.post("/orders", json={"product_id": 1, "quantity": quantity})
    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "quantity"] for error in response.json()["detail"]
    )


@pytest.mark.parametrize("product_id", [1.5, 1.0, "1", True, None])
def test_product_id_requires_integer(client, product_id):
    response = client.post("/orders", json={"product_id": product_id, "quantity": 1})
    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "product_id"] for error in response.json()["detail"]
    )


@pytest.mark.parametrize("body", [{}, {"product_id": 1}, {"quantity": 2}])
def test_missing_fields(client, body):
    response = client.post("/orders", json=body)
    assert response.status_code == 422
    missing = {error["loc"][-1] for error in response.json()["detail"]}
    assert missing == {"product_id", "quantity"} - body.keys()


@pytest.mark.parametrize(
    "body",
    ["", " ", '{"product_id":', "null", "[]", '{"product_id":1,"quantity":2} {}'],
)
def test_invalid_body(client, body):
    response = client.post(
        "/orders", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_unknown_field(client):
    response = client.post(
        "/orders", json={"product_id": 1, "quantity": 2, "discount": 10}
    )
    assert response.status_code == 422
    assert any(
        error["type"] == "extra_forbidden" and error["loc"] == ["body", "discount"]
        for error in response.json()["detail"]
    )


def test_unknown_route(client):
    response = client.get("/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_openapi(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["version"] == "1.0.0"
    assert set(schema["paths"]) == {
        "/live",
        "/ready",
        "/health",
        "/products",
        "/orders",
    }
    models = schema["components"]["schemas"]
    request = models["OrderRequest"]
    assert set(request["required"]) == {"product_id", "quantity"}
    assert request["additionalProperties"] is False
    assert request["properties"]["product_id"]["type"] == "integer"
    assert request["properties"]["quantity"]["type"] == "integer"
    assert request["properties"]["quantity"]["exclusiveMinimum"] == 0
    assert schema["paths"]["/orders"]["post"]["requestBody"]["required"] is True
    assert schema["paths"]["/orders"]["post"]["responses"]["201"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/OrderResponse"}
    assert {
        "HealthResponse",
        "LiveResponse",
        "Product",
        "OrderResponse",
    } <= models.keys()


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_documentation(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/openapi.json" in response.text
